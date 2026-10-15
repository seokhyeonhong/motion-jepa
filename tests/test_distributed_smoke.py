"""Two-process Gloo integration test for the public local-device launcher."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml

from _npy_fixture import write_npy_dataset


class DistributedTrainingSmokeTest(unittest.TestCase):
    def test_two_cpu_processes_train_through_main(self):
        self._run_training(context=False)

    def test_two_cpu_processes_with_context_warmup(self):
        self._run_training(context=True)

    def _run_training(self, *, context):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "output"
            write_npy_dataset(
                dataset,
                [np.full((4, 6), index, dtype=np.float32) for index in range(2)],
            )
            config = {
                "data": {
                    "batch_size": 1,
                    "root_path": str(dataset),
                    "meta_files": ["train.txt"],
                    "num_workers": 0,
                    "pin_mem": False,
                    "persistent_workers": False,
                    "drop_last": True,
                    "num_frames": 4,
                    "fps": 60,
                    "motion_dim": 6,
                    "num_joints": 30,
                    "normalize": False,
                },
                "logging": {
                    "folder": str(output),
                    "write_tag": "ddp-smoke",
                    "log_freq": 1,
                    "checkpoint_freq": 1,
                },
                "mask": {
                    "allow_overlap": False,
                    "num_enc_masks": 1,
                    "num_pred_masks": 1,
                    "enc_frame_mask_ratio": [0.75, 0.75],
                    "pred_frame_mask_ratio": [0.25, 0.25],
                },
                "meta": {
                    "seed": 0,
                    "load_checkpoint": False,
                    "model_name": "mot_tiny_1d",
                    "predictor_name": "mot_predictor_tiny_1d",
                    "pred_depth": 1,
                    "pred_emb_dim": 12,
                    "use_bfloat16": False,
                    "use_float16": False,
                },
                "optimization": {
                    "ema": [0.9, 1.0],
                    "epochs": 1,
                    "final_lr": 1.0e-5,
                    "final_weight_decay": 0.4,
                    "ipe_scale": 1.0,
                    "lr": 1.0e-3,
                    "start_lr": 1.0e-4,
                    "warmup": 0,
                    "weight_decay": 0.04,
                },
            }
            if context:
                config["loss"] = {"context": {"weight": .1, "warmup_epochs": 1,
                                               "distance_weighting": True}}
                config["position_encoding"] = {"temporal": "rope"}
                config["optimization"]["epochs"] = 2
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            environment = dict(os.environ)
            for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
                environment.pop(key, None)
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve().parents[1] / "main.py"),
                    "--fname",
                    str(config_path),
                    "--devices",
                    "cpu",
                    "cpu",
                    "--master-port",
                    str(port),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=90,
                env=environment,
            )
            self.assertTrue((output / "ddp-smoke-latest.pth.tar").is_file())
            if context:
                rows = [json.loads(line) for line in
                        (output / "training-epochs.jsonl").read_text().splitlines()]
                self.assertEqual([row["context_active_steps"] for row in rows], [0, 1])
                self.assertEqual([row["context_weight"] for row in rows], [0., .1])
                self.assertGreater(rows[1]["context_loss"], 0)
                self.assertAlmostEqual(rows[1]["loss"],
                    rows[1]["masked_loss"] + rows[1]["weighted_context_loss"], places=6)


if __name__ == "__main__":
    unittest.main()

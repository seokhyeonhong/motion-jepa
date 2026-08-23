"""End-to-end single-process training smoke test."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from _npy_fixture import write_npy_dataset
from train import main as train_main


class TrainingSmokeTest(unittest.TestCase):
    def test_one_epoch_1d_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            output = root / "output"
            write_npy_dataset(
                dataset,
                [
                    np.random.default_rng(index).normal(size=(4, 6)).astype(np.float32)
                    for index in range(2)
                ],
            )
            config = {
                "data": {
                    "batch_size": 2,
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
                    "stats_path": None,
                },
                "logging": {
                    "folder": str(output),
                    "write_tag": "smoke",
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
                    "read_checkpoint": None,
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
            result = train_main(config, device="cpu")
            self.assertEqual(result["global_step"], 1)
            self.assertTrue(Path(result["checkpoint"]).is_file())

            # Simulate a format-v1 checkpoint written before architecture
            # signatures were added; resume must derive the raw layout from config.
            checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)
            checkpoint.pop("architecture")
            torch.save(checkpoint, result["checkpoint"])

            config["meta"]["load_checkpoint"] = True
            config["optimization"]["epochs"] = 2
            resumed = train_main(config, device="cpu")
            self.assertEqual(resumed["global_step"], 2)
            self.assertEqual(resumed["next_epoch"], 2)


if __name__ == "__main__":
    unittest.main()

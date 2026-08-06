"""Launch dispatch tests for local devices and torchrun environments."""

from __future__ import annotations

import argparse
import os
import unittest
from unittest import mock

import main


def _args(devices):
    return argparse.Namespace(
        fname="config.yaml", devices=devices, debug=False, master_port=45678
    )


class MainLaunchTest(unittest.TestCase):
    def test_single_local_device_runs_directly(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(main, "_run", return_value="done") as run,
        ):
            self.assertEqual(main.launch(_args(["cuda:3"])), "done")
        run.assert_called_once_with("config.yaml", "cuda:3")

    def test_multiple_local_devices_spawn_one_process_each(self):
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch.object(main.mp, "spawn") as spawn,
        ):
            main.launch(_args(["cuda:1", "cuda:4"]))
        self.assertEqual(spawn.call_args.kwargs["nprocs"], 2)
        self.assertTrue(spawn.call_args.kwargs["join"])

    def test_torchrun_environment_uses_local_rank(self):
        environment = {"RANK": "3", "WORLD_SIZE": "8", "LOCAL_RANK": "1"}
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch.object(main.torch.cuda, "is_available", return_value=True),
            mock.patch.object(main, "_run", return_value="torchrun") as run,
        ):
            self.assertEqual(main.launch(_args(["cuda:7"])), "torchrun")
        run.assert_called_once_with("config.yaml", "cuda:1")


if __name__ == "__main__":
    unittest.main()

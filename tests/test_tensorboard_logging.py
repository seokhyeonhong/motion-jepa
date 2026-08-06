"""Tests for dependency-free TensorBoard metric emission."""

from __future__ import annotations

import unittest

from train import _write_tensorboard_interval


class _Writer:
    def __init__(self):
        self.scalars = {}
        self.flush_count = 0

    def add_scalar(self, name, value, step):
        self.scalars[name] = (value, step)

    def flush(self):
        self.flush_count += 1


class TensorBoardLoggingTest(unittest.TestCase):
    def test_interval_values_are_emitted_at_the_global_step(self):
        writer = _Writer()
        _write_tensorboard_interval(
            writer,
            global_step=100,
            epoch=2,
            loss=0.25,
            learning_rate=1.0e-4,
            weight_decay=0.04,
            time_ms=12.5,
            memory_mib=1024.0,
            grad_first=0.1,
            grad_last=0.2,
            grad_average=0.15,
        )
        self.assertEqual(writer.scalars["train/loss"], (0.25, 100))
        self.assertEqual(writer.scalars["train/iteration_time_ms"], (12.5, 100))
        self.assertEqual(writer.scalars["train/epoch"], (2.0, 100))
        self.assertEqual(writer.flush_count, 1)

    def test_none_writer_is_a_noop(self):
        _write_tensorboard_interval(
            None,
            global_step=1,
            epoch=1,
            loss=1.0,
            learning_rate=1.0,
            weight_decay=1.0,
            time_ms=1.0,
            memory_mib=0.0,
            grad_first=0.0,
            grad_last=0.0,
            grad_average=0.0,
        )


if __name__ == "__main__":
    unittest.main()

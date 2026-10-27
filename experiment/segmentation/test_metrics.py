import numpy as np
import pytest
import torch
from .metrics import FrameMetricAccumulator, segment_counts


@pytest.mark.parametrize("length,expected", [
    (1, [[1, 0, 0], [0, 1, 1], [0, 1, 1]]),
    (2, [[1, 0, 0], [0, 1, 1], [0, 1, 1]]),
    (3, [[1, 0, 0], [1, 0, 0], [0, 1, 1]]),
    (5, [[1, 0, 0], [1, 0, 0], [1, 0, 0]]),
])
def test_iou_thresholds(length, expected):
    truth = np.ones((10, 1), dtype=bool)
    prediction = np.zeros_like(truth)
    prediction[:length] = True
    counts = segment_counts(prediction, truth, np.ones(10, dtype=bool))
    assert counts[:, 0].tolist() == expected


def test_fragmentation_and_overlapping_classes():
    truth = np.ones((10, 2), dtype=bool)
    prediction = truth.copy()
    prediction[4:6, 0] = False
    counts = segment_counts(prediction, truth, np.ones(10, dtype=bool))
    assert counts[:, 0].tolist() == [[1, 1, 0], [1, 1, 0], [0, 2, 1]]
    assert counts[:, 1].tolist() == [[1, 0, 0]] * 3


def test_gaps_and_padding_break_intervals():
    truth = np.ones((8, 1), dtype=bool)
    mask = np.array([1, 1, 0, 1, 1, 0, 0, 0], dtype=bool)
    assert segment_counts(truth, truth, mask)[:, 0].tolist() == [[2, 0, 0]] * 3
    assert segment_counts(truth, ~truth, mask)[:, 0].tolist() == [[0, 2, 0]] * 3


def test_macro_and_window_batch_invariance():
    labels = torch.tensor([[[1, 0], [1, 0]], [[0, 0], [0, 0]]])
    logits = torch.tensor([[[1., -1.], [1., -1.]], [[1., -1.], [1., -1.]]])
    mask = torch.ones(2, 2, dtype=torch.bool)
    batched = FrameMetricAccumulator(2, [0])
    batched.update(logits, labels, mask)
    separate = FrameMetricAccumulator(2, [0])
    for l, y, m in zip(logits, labels, mask):
        separate.update(l, y, m)
    assert batched.compute() == separate.compute()
    result = batched.compute()["babel-120"]
    assert result["macro_f1"] == pytest.approx(1/3)
    assert result["segment_f1_50"] == pytest.approx(2/3)

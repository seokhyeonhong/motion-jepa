"""Exact threshold-grouped frame AP, independent of score ties and batch order.

The legacy clip classifier uses a different tie convention. These metrics do not
change that historical protocol. All valid 120-class frames remain negatives for
the 60-class subset even when their positive classes are outside that subset.
"""

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F


def binary_average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float | None:
    """Noninterpolated AP over unique thresholds; absent positives return None."""
    if scores.ndim != 1 or scores.shape != targets.shape or len(scores) == 0:
        raise ValueError("AP requires matching nonempty score and target vectors")
    if not torch.isfinite(scores).all() or not ((targets == 0) | (targets == 1)).all():
        raise ValueError("AP requires finite scores and binary targets")
    positives = int(targets.sum())
    if positives == 0:
        return None
    ranked_scores, order = torch.sort(scores.float(), descending=True)
    true_positives = targets[order].to(torch.float64).cumsum(0)
    # End of each tied group, rather than positive-by-positive ranks within it.
    ends = torch.cat((ranked_scores[1:] != ranked_scores[:-1],
                      torch.ones(1, dtype=torch.bool, device=scores.device)))
    indices = ends.nonzero(as_tuple=False).flatten()
    cumulative = true_positives[indices]
    increments = torch.diff(cumulative, prepend=cumulative.new_zeros(1))
    precision = cumulative / (indices.to(torch.float64) + 1)
    return float((increments * precision).sum() / positives)


def segment_counts(prediction, truth, supervised):
    """Per-class TP/FP/FN at IoU .10/.25/.50 for one multilabel window.

    Ignored frames break intervals. Predictions are matched chronologically to
    the maximum-IoU ground-truth interval; a ground-truth interval matches once.
    Different classes can overlap. Window boundaries never join segments.
    """
    counts = np.zeros((3, prediction.shape[-1], 3), dtype=np.int64)
    def intervals(active):
        edges = np.diff(np.pad(active.astype(np.int8), (1, 1)))
        return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    active_classes = ((prediction | truth) & supervised[:, None]).any(axis=0)
    for c in np.flatnonzero(active_classes):
        ps, pe = intervals(prediction[:, c] & supervised)
        gs, ge = intervals(truth[:, c] & supervised)
        for k, threshold in enumerate((.10, .25, .50)):
            matched = np.zeros(len(gs), dtype=bool)
            tp = 0
            for start, end in zip(ps, pe):
                if not len(gs):
                    continue
                intersection = np.maximum(0, np.minimum(end, ge) - np.maximum(start, gs))
                union = np.maximum(end, ge) - np.minimum(start, gs)
                iou = intersection / union
                best = int(iou.argmax())
                if iou[best] >= threshold and not matched[best]:
                    matched[best] = True
                    tp += 1
            counts[k, c] = tp, len(ps) - tp, len(gs) - tp
    return counts


class FrameMetricAccumulator:
    """Collect the complete validation split, with optional CUDA sorting."""

    def __init__(self, num_classes: int, class_indices_60: Sequence[int], *, device="cpu"):
        self.num_classes = int(num_classes)
        self.class_indices_60 = tuple(int(value) for value in class_indices_60)
        if not self.class_indices_60 or len(set(self.class_indices_60)) != len(self.class_indices_60):
            raise ValueError("Subset class indices must be nonempty and unique")
        if min(self.class_indices_60) < 0 or max(self.class_indices_60) >= self.num_classes:
            raise ValueError("Subset class index outside the full vocabulary")
        self.device = torch.device(device)
        self.scores = []
        self.targets = []
        self.segment_counts = np.zeros((3, self.num_classes, 3), dtype=np.int64)

    def update(self, logits: torch.Tensor, labels: torch.Tensor, supervised: torch.Tensor):
        if logits.shape != labels.shape or logits.shape[-1] != self.num_classes:
            raise ValueError("Frame targets must match logits [..., classes]")
        if supervised.shape != logits.shape[:-1]:
            raise ValueError("Supervision mask must match the frame axes")
        if logits.ndim not in (2, 3):
            raise ValueError("Segment metrics require [time, classes] or [batch, time, classes]")
        selected = supervised.bool()
        if selected.any():
            scores = logits[selected].detach().float().to(self.device)
            targets = labels[selected].detach().to(self.device)
            if not torch.isfinite(scores).all() or not ((targets == 0) | (targets == 1)).all():
                raise ValueError("Frame metrics require finite logits and binary targets")
            self.scores.append(scores)
            self.targets.append(targets.bool())
            predictions = (logits.detach() >= 0).cpu().numpy()
            truths = labels.detach().bool().cpu().numpy()
            masks = selected.cpu().numpy()
            if logits.ndim == 2:
                predictions, truths, masks = predictions[None], truths[None], masks[None]
            for prediction, truth, mask in zip(predictions, truths, masks):
                self.segment_counts += segment_counts(prediction, truth, mask)

    def compute(self) -> dict:
        if not self.scores:
            raise ValueError("No supervised frames in the validation split")
        scores, targets = torch.cat(self.scores), torch.cat(self.targets)
        ap = [binary_average_precision(scores[:, c], targets[:, c])
              for c in range(self.num_classes)]
        result = {}
        for name, columns in (("babel-120", tuple(range(self.num_classes))),
                              ("babel-60", self.class_indices_60)):
            index = torch.tensor(columns, device=self.device)
            logits, truth = scores.index_select(1, index), targets.index_select(1, index)
            prediction = logits >= 0.0  # sigmoid(logits) >= 0.5, without quantization.
            tp = int((prediction & truth).sum())
            fp = int((prediction & ~truth).sum())
            fn = int((~prediction & truth).sum())
            class_tp = (prediction & truth).sum(0).double()
            class_fp = (prediction & ~truth).sum(0)
            class_fn = (~prediction & truth).sum(0)
            macro_f1 = (2 * class_tp / (2 * class_tp + class_fp + class_fn).clamp_min(1)).mean()
            segments = self.segment_counts[:, list(columns)].sum(axis=1)
            present = [ap[c] for c in columns if ap[c] is not None]
            result[name] = {
                "frame_map": sum(present) / len(present) if present else None,
                "bce": float(F.binary_cross_entropy_with_logits(logits, truth.float())),
                "micro_f1": 2 * tp / max(2 * tp + fp + fn, 1),
                "macro_f1": float(macro_f1),
                **{f"segment_f1_{threshold}": float(2 * stp / max(2 * stp + sfp + sfn, 1))
                   for threshold, (stp, sfp, sfn) in zip((10, 25, 50), segments)},
                "classes_with_positives": len(present),
                "classes_without_positives": len(columns) - len(present),
                "supervised_frames": len(scores),
            }
        return result

"""Frame-token 1D MotionJEPA models."""

from .motion_transformer_1d import (
    MotionTransformer1D,
    MotionTransformerPredictor1D,

    mot_tiny_1d,
    mot_small_1d,
    mot_base_1d,
    mot_large_1d,
    mot_huge_1d,
    mot_giant_1d,

    mot_predictor_tiny_1d,
    mot_predictor_small_1d,
    mot_predictor_base_1d,
    mot_predictor_large_1d,
    mot_predictor_huge_1d,
    mot_predictor_giant_1d,
)
from .token_layout import TokenLayout

MODEL_FACTORIES = {
    "mot_tiny_1d": mot_tiny_1d,
    "mot_small_1d": mot_small_1d,
    "mot_base_1d": mot_base_1d,
    "mot_large_1d": mot_large_1d,
    "mot_huge_1d": mot_huge_1d,
    "mot_giant_1d": mot_giant_1d,
}

PREDICTOR_FACTORIES = {
    "mot_predictor_tiny_1d": mot_predictor_tiny_1d,
    "mot_predictor_small_1d": mot_predictor_small_1d,
    "mot_predictor_base_1d": mot_predictor_base_1d,
    "mot_predictor_large_1d": mot_predictor_large_1d,
    "mot_predictor_huge_1d": mot_predictor_huge_1d,
    "mot_predictor_giant_1d": mot_predictor_giant_1d,
}

__all__ = [
    "MotionTransformer1D",
    "MotionTransformerPredictor1D",
    "TokenLayout",
    "MODEL_FACTORIES",
    "PREDICTOR_FACTORIES",
    *MODEL_FACTORIES,
    *PREDICTOR_FACTORIES,
]

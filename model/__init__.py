"""Public Motion-JEPA model variants and named factories."""

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
from .motion_transformer_2d import (
    MotionFeatureTokenizer2D,
    MotionTransformer2D,
    MotionTransformerPredictor2D,
    mot_tiny_2d,
    mot_small_2d,
    mot_base_2d,
    mot_large_2d,
    mot_huge_2d,
    mot_giant_2d,
    mot_predictor_tiny_2d,
    mot_predictor_small_2d,
    mot_predictor_base_2d,
    mot_predictor_large_2d,
    mot_predictor_huge_2d,
    mot_predictor_giant_2d,
)

MODEL_FACTORIES = {
    factory.__name__: factory
    for factory in (
        mot_tiny_1d,
        mot_small_1d,
        mot_base_1d,
        mot_large_1d,
        mot_huge_1d,
        mot_giant_1d,

        mot_tiny_2d,
        mot_small_2d,
        mot_base_2d,
        mot_large_2d,
        mot_huge_2d,
        mot_giant_2d,
    )
}

PREDICTOR_FACTORIES = {
    factory.__name__: factory
    for factory in (
        mot_predictor_tiny_1d,
        mot_predictor_small_1d,
        mot_predictor_base_1d,
        mot_predictor_large_1d,
        mot_predictor_huge_1d,
        mot_predictor_giant_1d,

        mot_predictor_tiny_2d,
        mot_predictor_small_2d,
        mot_predictor_base_2d,
        mot_predictor_large_2d,
        mot_predictor_huge_2d,
        mot_predictor_giant_2d,
    )
}

__all__ = [
    "MODEL_FACTORIES",
    "MotionFeatureTokenizer2D",
    "MotionTransformer1D",
    "MotionTransformer2D",
    "MotionTransformerPredictor1D",
    "MotionTransformerPredictor2D",
    *MODEL_FACTORIES,
    *PREDICTOR_FACTORIES,
]

"""Named Motion-JEPA encoder size specifications."""

MODEL_SPECS = {
    "tiny": {"embed_dim": 192, "depth": 6, "num_heads": 3},
    "small": {"embed_dim": 384, "depth": 8, "num_heads": 6},
    "base": {"embed_dim": 768, "depth": 12, "num_heads": 12},
    "large": {"embed_dim": 1024, "depth": 24, "num_heads": 16},
    "huge": {"embed_dim": 1280, "depth": 32, "num_heads": 16},
    "giant": {"embed_dim": 1408, "depth": 48, "num_heads": 16},
}

__all__ = ["MODEL_SPECS"]

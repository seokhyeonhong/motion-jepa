"""Shared experiment I/O, reproducibility, and device utilities."""

from __future__ import annotations

import functools
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class Metrics:
    loss: float
    top1_accuracy: float
    macro_accuracy: float
    top5_accuracy: float


def _sha256_file(path: Path) -> str:
    stat = path.stat()
    return _sha256_file_version(path, stat.st_size, stat.st_mtime_ns)


@functools.lru_cache(maxsize=None)
def _sha256_file_version(path: Path, size: int, mtime_ns: int) -> str:
    del size, mtime_ns
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_save(value: Any, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {device}")
        torch.cuda.set_device(device)
    return device


def _torch_load_checkpoint(path: Path) -> dict[str, Any]:
    """Load checkpoints written with NumPy 1.x or 2.x module names."""
    aliases: dict[str, Any] = {}
    try:
        import numpy._core  # type: ignore[attr-defined]  # noqa: F401
    except ModuleNotFoundError:
        module_aliases = {
            "numpy._core": np.core,
            "numpy._core.multiarray": np.core.multiarray,
            "numpy._core.numeric": np.core.numeric,
        }
        for name, module in module_aliases.items():
            if name not in sys.modules:
                sys.modules[name] = module
                aliases[name] = module
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    finally:
        for name, module in aliases.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint payload must be a mapping: {path}")
    return checkpoint

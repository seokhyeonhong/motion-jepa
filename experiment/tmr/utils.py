"""Small shared helpers for the standalone alignment experiment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from experiment.utils import resolve_device as _resolve_device


def resolve_device(value: str) -> torch.device:
    # torch.cuda.set_device requires an index even though torch.device accepts
    # the common CLI shorthand "cuda".
    return _resolve_device("cuda:0" if value == "cuda" else value)



def atomic_torch_save(value: Any, path: Path) -> None:
    """Flush checkpoint contents and rename before publishing the new state."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    if hasattr(os, "O_DIRECTORY"):
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

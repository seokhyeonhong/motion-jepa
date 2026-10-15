"""Streaming frozen-backbone caches; training reads only completed cache files."""

from __future__ import annotations

import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset import MotionDataset
from experiment.utils import (
    _atomic_json_save, _sha256_file,
)
from .utils import resolve_device
from .dataset import (
    DEFAULT_ANNOTATIONS_PATH, DEFAULT_CACHE_ROOT, DEFAULT_DATASET_ROOT,
    DEFAULT_TEXT_MODEL, DEFAULT_TEXT_ADAPTER, SPLITS, PreparedPairDataset, RaggedTokenBank,
    build_paired_index, json_digest, select_training_subset,
)


CACHE_FORMAT_VERSION = 2


def _model_source_signature(model: str) -> dict[str, Any]:
    path = Path(model).expanduser()
    if not path.is_dir():
        return {"model": model, "local_files": None}
    path = path.resolve()
    files = sorted(
        candidate for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix in {".json", ".safetensors", ".bin", ".model"}
    )
    return {"model": str(path), "local_files": {file.name: _sha256_file(file) for file in files}}


def _text_signature(paired, model, revision, max_length, adapter=None, adapter_revision=None):
    return {
        "format_version": CACHE_FORMAT_VERSION, "dtype": "bfloat16_uint16",
        "dataset": paired["provenance"], "catalog_sha256": paired["catalog_sha256"],
        "model_source": _model_source_signature(str(model)),
        "text_revision": revision, "max_text_length": int(max_length),
        "feature": "llm2vec_sentence_mean_bidirectional_v1",
        "adapter_source": _model_source_signature(str(adapter or DEFAULT_TEXT_ADAPTER)),
        "adapter_revision": adapter_revision,
        "extraction_batch_size": 1, "doc_max_length": 400,
    }


def _stats_signature(root: Path):
    return {
        "stats_root": str(root.resolve()),
        "stats_mean_sha256": _sha256_file(root / "mean.npy"),
        "stats_std_sha256": _sha256_file(root / "std.npy"),
    }


def _existing_bank(path, signature, recompute):
    if recompute:
        return None
    if path.exists():
        return RaggedTokenBank(path, signature)
    return None


def write_ragged_bank(path, keys, lengths, feature_dim, signature, batches, extra=None):
    """Write bounded batches into a BF16 memmap and publish completion last."""
    path = Path(path)
    if len(keys) != len(lengths) or any(int(length) < 1 for length in lengths):
        raise ValueError("Token-bank keys and strictly positive lengths must agree")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir()
    offsets = np.concatenate(([0], np.cumsum(np.asarray(lengths, dtype=np.int64))))
    values = np.lib.format.open_memmap(
        temporary / "values.npy", mode="w+", dtype=np.uint16,
        shape=(int(offsets[-1]), int(feature_dim)),
    )
    np.save(temporary / "offsets.npy", offsets, allow_pickle=False)
    cursor = 0
    try:
        for sequences in batches:
            for sequence in sequences:
                expected = (int(lengths[cursor]), int(feature_dim))
                if tuple(sequence.shape) != expected or not torch.isfinite(sequence).all():
                    raise ValueError(f"Invalid token-cache output: expected {expected}, got {tuple(sequence.shape)}")
                bits = sequence.detach().to(device="cpu", dtype=torch.bfloat16).contiguous().view(torch.uint16).numpy()
                values[int(offsets[cursor]):int(offsets[cursor + 1])] = bits
                cursor += 1
        if cursor != len(keys):
            raise ValueError("Token extraction did not cover every cache row")
        values.flush()
        del values
        marker = {
            "signature": signature, "keys": list(keys), "feature_dim": int(feature_dim),
            "num_tokens": int(offsets[-1]), "max_length": max(lengths, default=0),
            "model_info": extra or {},
        }
        _atomic_json_save(marker, temporary / "complete.json")
        if path.exists():
            shutil.rmtree(path)
        os.replace(temporary, path)
    except BaseException:
        # A failed extraction must never look complete to a training reader.
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return RaggedTokenBank(path, signature)


def load_text_backbone(model_name, revision, device, adapter=None, adapter_revision=None):
    """Load the frozen LLM2Vec backbone; training only needs its cached features."""
    adapter = str(adapter or DEFAULT_TEXT_ADAPTER)
    try:
        from .llm2vec import load_llm2vec
        model, tokenizer, info = load_llm2vec(
            model_name, revision, adapter, adapter_revision, device,
        )
    except ImportError as error:
        raise RuntimeError("Install text dependencies: pip install -r experiment/tmr/requirements.txt") from error
    except OSError as error:
        raise RuntimeError(
            "Cannot load LLM2Vec. Check the MNTP/supervised adapter paths and "
            "Hugging Face access to meta-llama/Meta-Llama-3-8B-Instruct."
        ) from error
    info["tokenizer_sha256"] = json_digest({
        "backend": tokenizer.backend_tokenizer.to_str(),
        "special_tokens": {key: str(value) for key, value in tokenizer.special_tokens_map.items()},
    })
    return model, tokenizer, info


def _prepare_text(path, paired, *, model_name, revision, max_length, device, batch_size, recompute, adapter=None, adapter_revision=None):
    signature = _text_signature(paired, model_name, revision, max_length, adapter, adapter_revision)
    existing = _existing_bank(path, signature, recompute)
    if existing is not None:
        return existing
    model, tokenizer, info = load_text_backbone(model_name, revision, device, adapter, adapter_revision)
    position_limit = info.get("max_position_embeddings")
    if position_limit is not None and max_length > position_limit:
        raise ValueError(
            f"--max-text-length {max_length} exceeds {model_name!r}'s positional "
            f"limit of {position_limit}"
        )
    keys = list(paired["catalog"])
    texts = [paired["catalog"][key] for key in keys]
    info.update(
        max_text_length=max_length,
        truncated_captions=sum(model.token_length(text, max_length) > max_length for text in texts),
    )

    def sentence_batches():
        for start in tqdm(range(0, len(texts), batch_size), desc="Cache LLM2Vec sentences"):
            yield model.encode(texts[start:start + batch_size], max_length, device)
    try:
        return write_ragged_bank(
            path, keys, [1] * len(keys), 4096, signature, sentence_batches(), info,
        )
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


class _MotionRows(Dataset):
    def __init__(self, base, records):
        self.base, self.records = base, records
        rows = {entry.sample_id: index for index, entry in enumerate(base.entries)}
        self.rows = [rows[record["sample_id"]] for record in records]
        for row, record in zip(self.rows, records):
            entry = base.entries[row]
            if entry.length != record["length"] or entry.fps != record["fps"] or entry.path != base.root_path / record["motion_path"]:
                raise ValueError(f"Paired interval/path disagrees with raw split index: {record['sample_id']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.base[self.rows[index]]


def _raw_dataset(root, split, paired, stats_root):
    provenance = paired["provenance"]
    return MotionDataset(
        root, f"{split}.txt", provenance["num_frames"], provenance["fps"],
        motion_dim=provenance["raw_motion_dim"], normalize=True, stats_path=stats_root,
    )


def prepare_caches(
    *, dataset_root=DEFAULT_DATASET_ROOT, annotations_path=DEFAULT_ANNOTATIONS_PATH,
    cache_root=DEFAULT_CACHE_ROOT, stats_path=None, text_model=DEFAULT_TEXT_MODEL,
    text_revision=None, text_adapter=None, text_adapter_revision=None, max_text_length=256, device="auto", text_batch_size=8, recompute_features=False, max_samples_per_split=None,
):
    if min(max_text_length, text_batch_size) < 1:
        raise ValueError("Lengths and batch sizes must be positive")
    root, cache = Path(dataset_root).expanduser().resolve(), Path(cache_root).expanduser().resolve()
    target_device = resolve_device(str(device))
    paired = build_paired_index(root, annotations_path, max_samples_per_split)
    if not paired["split_counts"]["train"]:
        raise ValueError("TMR has no captioned training samples")
    stats_root = Path(stats_path).expanduser().resolve() if stats_path is not None else root / "stats"
    stats = _stats_signature(stats_root)
    raw_datasets = {split: _raw_dataset(root, split, paired, stats_root) for split in SPLITS}
    for split, base in raw_datasets.items():
        records = [record for record in paired["records"] if record["split"] == split]
        _MotionRows(base, records)
    # Validate motion/statistics before an expensive text-model download or run.
    text_bank = _prepare_text(
        cache / "text", paired, model_name=str(text_model), revision=text_revision,
        max_length=max_text_length, device=target_device,
        batch_size=text_batch_size, recompute=recompute_features,
        adapter=text_adapter, adapter_revision=text_adapter_revision,
    )
    metadata = {
        "format_version": CACHE_FORMAT_VERSION, "input_source": "raw",
        "motion_dim": paired["provenance"]["raw_motion_dim"],
        "motion_num_tokens": paired["provenance"]["num_frames"],
        "text_dim": text_bank.metadata["feature_dim"],
        "text_source": text_bank.metadata["model_info"], "text_signature": text_bank.metadata["signature"],
        **stats,
        "paired_index_sha256": paired["paired_index_sha256"], "catalog_sha256": paired["catalog_sha256"],
        "split_counts": paired["split_counts"], "filtered_counts": paired["filtered_counts"],
        "provenance": paired["provenance"],
    }
    directory = cache / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_json_save(paired, directory / "paired-index.json")
    _atomic_json_save(metadata, directory / "prepared.json")
    return metadata


def load_prepared_datasets(
    dataset_root, annotations_path, cache_root, stats_path=None,
    text_model=DEFAULT_TEXT_MODEL, text_revision=None, max_text_length=256,
    train_fraction=1.0, train_subset_seed=42, text_adapter=None, text_adapter_revision=None,
):
    """Validate provenance without instantiating or deserializing a backbone."""
    root, cache = Path(dataset_root).expanduser().resolve(), Path(cache_root).expanduser().resolve()
    marker = cache / "raw" / "prepared.json"
    if not marker.is_file():
        raise FileNotFoundError(f"Prepared TMR cache is missing: {marker}; run prepare_cache")
    metadata = json.loads(marker.read_text())
    if metadata.get("format_version") != CACHE_FORMAT_VERSION or metadata.get("input_source") != "raw":
        raise ValueError(
            "Unsupported prepared TMR cache format; rebuild with prepare_cache "
            "--recompute-features to cache individual caption candidates"
        )
    paired = build_paired_index(root, annotations_path, metadata["provenance"]["max_samples_per_split"])
    for key in ("paired_index_sha256", "catalog_sha256", "provenance", "split_counts", "filtered_counts"):
        if metadata[key] != paired[key]:
            raise ValueError("Prepared TMR cache metadata is stale; use --recompute-features")
    train_records, subset = select_training_subset(
        [record for record in paired["records"] if record["split"] == "train"],
        train_fraction, train_subset_seed,
    )
    signature = _text_signature(paired, text_model, text_revision, max_text_length, text_adapter, text_adapter_revision)
    text_bank = RaggedTokenBank(cache / "text", signature)
    if metadata["text_signature"] != signature or metadata["text_source"] != text_bank.metadata["model_info"]:
        raise ValueError("Prepared text cache metadata is stale; use --recompute-features")
    stats_root = Path(stats_path).expanduser().resolve() if stats_path is not None else root / "stats"
    stats = _stats_signature(stats_root)
    if any(metadata[key] != value for key, value in stats.items()):
        raise ValueError("Prepared normalization statistics are stale; use --recompute-features")
    if subset is not None:
        metadata["train_subset"] = subset
    datasets = {}
    for split in SPLITS:
        records = train_records if split == "train" else [record for record in paired["records"] if record["split"] == split]
        datasets[split] = PreparedPairDataset(
            records, text_bank, raw_dataset=_raw_dataset(root, split, paired, stats_root),
            sample_captions=split == "train",
        )
    return datasets, metadata


__all__ = ["prepare_caches", "load_prepared_datasets", "load_text_backbone", "write_ragged_bank"]

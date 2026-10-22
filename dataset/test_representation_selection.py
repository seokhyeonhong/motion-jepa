"""Representation selection survives preprocessing, loading and JEPA resume."""
import copy
import importlib
import json

import numpy as np
import pytest
import torch

from dataset.motion_dataset import MotionDataset
from dataset import preprocess_bones_seed as preprocessing
from motion_rep.registry import get_representation, validate_representation_output


def make_dataset(root, name):
    args = preprocessing.parse_args(["--representation", name, "--output", str(root)])
    args.num_frames, args.min_frames = 12, 1
    root.mkdir()
    records = {split: [] for split in preprocessing.SPLITS}
    dim = get_representation(name).FEATURE_DIM
    rng = np.random.default_rng(42)
    for split in records:
        for i in range(2):
            record = {"id": f"clip{i}", "split": split, "fps": 30, "length": 12,
                      "motion_dim": dim, "motion": rng.normal(size=(12, dim)).astype(np.float32)}
            records[split].append(preprocessing._save_record_motion(record, root))
    preprocessing.finalize_processed_dataset(
        args, records, [], {split: 2 for split in records},
        source_dataset="fixture", segmentation=preprocessing.SEGMENTATION_POLICY,
        metadata_extra={"preprocessing_protocol": preprocessing.preprocessing_protocol(args),
                        "preprocessing_protocol_sha256": preprocessing._protocol_sha256(
                            preprocessing.preprocessing_protocol(args))},
    )
    return args


@pytest.mark.parametrize("name", ["MotionJEPAMotionRep", "LocalMotionJEPARep"])
def test_finalize_load_and_mismatch(tmp_path, name):
    args = make_dataset(tmp_path / "data", name)
    rep = get_representation(name)
    loaded = MotionDataset(args.output, "train.txt", 12, 30, normalize=True, representation=name)
    assert loaded[0][0].shape == (12, rep.FEATURE_DIM)
    assert np.isfinite(loaded[0][0]).all()
    assert json.loads((args.output / "meta.json").read_text())["representation"] == rep.FORMAT_NAME
    validate_representation_output(args)
    other = "MotionJEPAMotionRep" if name == "LocalMotionJEPARep" else "LocalMotionJEPARep"
    with pytest.raises(ValueError, match="representation"):
        MotionDataset(args.output, "train.txt", 12, 30, representation=other)
    args.representation = other
    with pytest.raises(ValueError, match="another representation"):
        validate_representation_output(args)


@pytest.mark.parametrize("module,initargs", [
    ("bones_seed", (".", ".", 12, 30, 12, 1)),
    ("100style", (".", ".", 30)),
    ("amass", (".", ".", 12, 30, 12)),
    ("babel", (".", ".", 1)),
    ("babel_segmentation", (".", ".", ())),
])
def test_worker_selection(module, initargs):
    module = importlib.import_module("dataset.preprocess_" + module)
    module._init_worker(*initargs, "LocalMotionJEPARep")
    assert module._REPRESENTATION.FEATURE_DIM == 365
    module._init_worker(*initargs, "MotionJEPAMotionRep")
    assert module._REPRESENTATION.FEATURE_DIM == 366


def test_local_training_online_metrics_and_resume(tmp_path, monkeypatch):
    import train
    from model import MODEL_FACTORIES, PREDICTOR_FACTORIES
    from model.test_centering_sharpening import _config, _small_encoder, _small_predictor
    make_dataset(tmp_path / "data", "LocalMotionJEPARep")
    monkeypatch.setitem(MODEL_FACTORIES, "mot_tiny_1d", _small_encoder)
    monkeypatch.setitem(PREDICTOR_FACTORIES, "mot_predictor_tiny_1d", _small_predictor)
    config = _config(tmp_path / "run", False)
    config["data"].update(root_path=str(tmp_path / "data"), representation="LocalMotionJEPARep",
                          normalize=True, num_workers=0)
    del config["data"]["motion_dim"]
    config["optimization"]["iterations"] = 3
    result = train.main(config, device="cpu")
    assert result["global_step"] == 3
    saved = torch.load(result["checkpoint"], weights_only=False)
    assert saved["config"]["data"]["motion_dim"] == 365
    resumed = copy.deepcopy(config)
    resumed["logging"]["folder"] = str(tmp_path / "resumed")
    resumed["meta"].update(load_checkpoint=True, read_checkpoint=str(tmp_path / "run/test-it2.pth.tar"))
    assert train.main(resumed, device="cpu")["global_step"] == 3
    bad = copy.deepcopy(config)
    bad["data"]["motion_dim"] = 366
    with pytest.raises(ValueError, match="requires motion_dim=365"):
        train.main(bad, device="cpu")

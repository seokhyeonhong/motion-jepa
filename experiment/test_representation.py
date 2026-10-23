"""Both downstream tasks accept local features and reject incompatible checkpoints."""
import json

import numpy as np
import pytest
import torch

from dataset.test_representation_selection import make_dataset
from experiment.representation import validate_dataset
from experiment.probing import FrozenFrameExtractor
from experiment.tmr import features, model as tmr_model, train as tmr_train, prepare_cache
from experiment.segmentation import train as seg_train
from experiment.segmentation.model import MotionSegmentationTransformer
from model import motion_transformer_1d
from motion_rep.registry import get_representation


@pytest.mark.parametrize("name", ["MotionJEPAMotionRep", "LocalMotionJEPARep"])
def test_downstream_raw_inputs_and_models(tmp_path, monkeypatch, name):
    torch.set_num_threads(1)
    root = tmp_path / "data"
    make_dataset(root, name)
    rep = validate_dataset(root, name)
    paired = {"provenance": {"num_frames": 12, "fps": 30,
                             "raw_motion_dim": rep.FEATURE_DIM,
                             "representation": rep.FORMAT_NAME}}
    data = features._raw_dataset(root, "train", paired, root / "stats")
    motion, fps, length = data[0]
    assert motion.shape == (12, rep.FEATURE_DIM)
    motion = torch.from_numpy(motion)[None]
    valid = torch.arange(12)[None] < length
    specs = {"embed_dim": 12, "depth": 1, "num_heads": 3}
    monkeypatch.setitem(tmr_model.MODEL_SPECS, "tiny", specs)
    retrieval = tmr_model.TextMotionAlignment(
        tmr_model.AlignmentConfig(8, rep.FEATURE_DIM, 12, 30, model_size="tiny"))
    output = retrieval.encode_motion(motion, valid)
    assert output.shape == (1, 256) and torch.isfinite(output).all()
    segmentation = MotionSegmentationTransformer("tiny", input_dim=rep.FEATURE_DIM,
                                                 num_classes=120, num_frames=12)
    logits = segmentation(motion, valid, torch.tensor([fps]))
    assert logits.shape == (1, 12, 120) and torch.isfinite(logits).all()
    logits.square().mean().backward()
    other = "LocalMotionJEPARep" if name == "MotionJEPAMotionRep" else "MotionJEPAMotionRep"
    with pytest.raises(ValueError, match="representation"):
        validate_dataset(root, other)
    # Reject before touching a text cache or loading an LLM.
    with pytest.raises(ValueError, match="representation"):
        features.load_prepared_datasets(root, root / "unused", root / "cache", representation=other)


@pytest.mark.parametrize("name", ["MotionJEPAMotionRep", "LocalMotionJEPARep"])
def test_frozen_representation_and_input_statistics(tmp_path, monkeypatch, name):
    torch.set_num_threads(1)
    rep = get_representation(name)
    def factory(**kwargs):
        return motion_transformer_1d.MotionTransformer1D(
            **kwargs, embed_dim=12, depth=1, num_heads=3)
    monkeypatch.setattr(motion_transformer_1d, "mot_tiny_1d", factory)
    model = factory(in_chans=rep.FEATURE_DIM, num_frames=4).eval()
    stats = tmp_path / "stats"
    stats.mkdir()
    np.save(stats / "mean.npy", np.ones(rep.FEATURE_DIM, dtype=np.float32))
    np.save(stats / "std.npy", np.full(rep.FEATURE_DIM, 2, dtype=np.float32))
    data = {"motion_dim": rep.FEATURE_DIM, "num_frames": 4, "normalize": True,
            "root_path": str(tmp_path)}
    if name == "LocalMotionJEPARep":
        data["representation"] = name
    path = tmp_path / "checkpoint.pt"
    torch.save({"config": {"meta": {"model_name": "mot_tiny_1d"}, "data": data},
                "target_encoder": model.state_dict()}, path)
    extractor = FrozenFrameExtractor(path, 1, representation=rep.FORMAT_NAME)
    raw = torch.randn(2, 4, rep.FEATURE_DIM)
    lengths = torch.tensor([2, 4])
    valid = torch.arange(4)[None] < lengths[:, None]
    fps = torch.tensor([30., 30.])
    with torch.no_grad():
        expected = model(((raw - 1) / 2).masked_fill(~valid[..., None], 0),
                         fps, valid_frames=valid, feature_layer=1)
    torch.testing.assert_close(extractor(raw, fps, lengths), expected)
    extractor.release()
    other = "LocalMotionJEPARep" if name == "MotionJEPAMotionRep" else "MotionJEPAMotionRep"
    with pytest.raises(ValueError, match="checkpoint representation"):
        FrozenFrameExtractor(path, 1, representation=other)


def test_cli_representation_selection():
    for module in (tmr_train, prepare_cache, seg_train):
        assert module.build_parser().parse_args([]).representation is None
        assert module.build_parser().parse_args(
            ["--representation", "LocalMotionJEPARep"]).representation == "LocalMotionJEPARep"


def test_local_raw_attentive_probes():
    from experiment.probing import AttentiveProbe
    from experiment.segmentation.model import MotionSegmentationProbe
    raw = torch.randn(2, 4, 365)
    valid = torch.tensor([[True, True, False, False], [True] * 4])
    retrieval = AttentiveProbe(365, 256, torch.zeros(365), torch.ones(365),
                               num_heads=5, pool=True)
    segmentation = MotionSegmentationProbe(365, 120, 4, torch.zeros(365),
                                           torch.ones(365), num_heads=5)
    assert retrieval(raw, valid).shape == (2, 256)
    assert segmentation(raw, valid, torch.tensor([30, 30])).shape == (2, 4, 120)


def test_segmentation_describe_local(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["segmentation", "--describe", "--model-size", "tiny",
                                   "--dataset-root", str(tmp_path / "absent"),
                                   "--representation", "LocalMotionJEPARep"])
    seg_train.main()
    assert json.loads(capsys.readouterr().out)["architecture"]["input_dim"] == 365


def test_local_defaults_and_explicit_paths(tmp_path):
    from experiment.representation import select_defaults
    for module, task in [(tmr_train, "tmr"), (prepare_cache, "tmr"),
                          (seg_train, "segmentation")]:
        original = module.build_parser().parse_args([])
        root = original.dataset_root
        select_defaults(original, task)
        assert original.dataset_root == root
        local = module.build_parser().parse_args(["--representation", "LocalMotionJEPARep"])
        select_defaults(local, task)
        assert "local-from366" in str(local.dataset_root)
        if task == "tmr":
            assert local.cache_root.name == "tmr-cache-llm2vec-local"
        custom = module.build_parser().parse_args(
            ["--representation", "LocalMotionJEPARep", "--dataset-root", str(tmp_path)])
        select_defaults(custom, task)
        assert custom.dataset_root == tmp_path

"""Processed-feature conversion and dataset serialization checks."""
import json

import numpy as np
import pytest
import torch

from dataset.convert_to_local import convert_dataset
from dataset.motion_dataset import MotionDataset
from dataset.test_representation_selection import make_dataset
from motion_rep import LocalMotionJEPARep, MotionJEPAMotionRep
from motion_rep.geometry import y_rotation


def motion(frames=12, scale=1):
    rotations = torch.eye(3, dtype=torch.float64).repeat(frames, 30, 1, 1)
    rotations[:, 0] = y_rotation(torch.linspace(2.8, 3.6, frames, dtype=torch.float64))
    rotations[:, 5] = y_rotation(torch.linspace(0, .4, frames, dtype=torch.float64))
    roots = torch.zeros(frames, 3, dtype=torch.float64)
    roots[:, 0] = torch.linspace(0, 1, frames) * scale
    roots[:, 1] = 1
    roots[:, 2] = torch.linspace(1, 2, frames) * scale
    return rotations, roots


@pytest.mark.parametrize("frames", [1, 12])
def test_feature_conversion_matches_direct(frames):
    rotations, roots = motion(frames)
    original = MotionJEPAMotionRep()(rotations, roots, to_canonicalize=False).float()
    rep = LocalMotionJEPARep()
    converted = rep.from_global_features(original.double())
    direct = rep(rotations, roots)
    torch.testing.assert_close(converted, direct, atol=1e-5, rtol=1e-5)
    batched = rep.from_global_features(original.double().repeat(2, 1, 1))
    torch.testing.assert_close(batched[0], converted)
    assert torch.equal(converted[:, -4:], original[:, -4:].double())


@pytest.mark.parametrize("workers", [1, 2])
def test_dataset_conversion_statistics_and_source_untouched(tmp_path, workers):
    source, output = tmp_path / "original", tmp_path / "local"
    make_dataset(source, "MotionJEPAMotionRep")
    expected = {}
    original_rep, rep = MotionJEPAMotionRep(), LocalMotionJEPARep()
    for split, scale in (("train", 1), ("val", 10), ("test", 20)):
        for entry in MotionDataset(source, f"{split}.txt", 12, 30).entries:
            rotations, roots = motion(scale=scale)
            original = original_rep(rotations, roots, to_canonicalize=False).float().numpy()
            np.save(entry.path, original)
            expected[entry.path.relative_to(source)] = rep.from_global_features(
                torch.from_numpy(original).double()).float().numpy()
    before = {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()}
    metadata = convert_dataset(source, output, workers=workers)
    assert metadata["motion_dim"] == 365
    for relative, values in expected.items():
        np.testing.assert_array_equal(np.load(output / relative), values)
    for relative, content in before.items():
        assert (source / relative).read_bytes() == content
    train = np.concatenate([a for p, a in expected.items() if p.parts[1] == "train"]).astype(np.float64)
    np.testing.assert_allclose(np.load(output / "stats/mean.npy"), train.mean(0), atol=1e-6)
    expected_std = np.sqrt(np.maximum(train.var(0), 1e-12))
    expected_std = np.where(expected_std < 1e-6, 1, expected_std)
    np.testing.assert_allclose(np.load(output / "stats/std.npy"), expected_std, atol=1e-6)
    for split in ("train", "val", "test"):
        assert (source / f"{split}.txt").read_bytes() == (output / f"{split}.txt").read_bytes()
        ds = MotionDataset(output, f"{split}.txt", 12, 30, normalize=True,
                           representation="LocalMotionJEPARep")
        assert ds[0][0].shape == (12, 365)
    with pytest.raises(FileExistsError):
        convert_dataset(source, output)
    with pytest.raises(ValueError, match="non-nested"):
        convert_dataset(source, source / "nested")
    with pytest.raises(ValueError, match="original"):
        convert_dataset(output, tmp_path / "invalid")


def test_failed_conversion_has_no_completion_metadata(tmp_path):
    source, output = tmp_path / "original", tmp_path / "local"
    make_dataset(source, "MotionJEPAMotionRep")
    # Fixture's random vectors are deliberately not valid unit heading features.
    with pytest.raises(RuntimeError, match="unit heading"):
        convert_dataset(source, output, workers=1)
    assert not (output / "meta.json").exists()
    assert "unit heading" in (output / "conversion-error.txt").read_text()

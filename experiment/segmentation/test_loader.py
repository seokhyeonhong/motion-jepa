import itertools
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from .train import shuffled_training_batches


def test_shuffled_passes_preserve_drop_last():
    generator = torch.Generator().manual_seed(42)
    batches = shuffled_training_batches(11, 4, generator)
    expected = torch.Generator().manual_seed(42)
    for _ in range(3):
        order = torch.randperm(11, generator=expected).tolist()
        assert next(batches) == order[:4]
        assert next(batches) == order[4:8]


def test_too_small_dataset():
    with pytest.raises(ValueError):
        next(shuffled_training_batches(3, 4, torch.Generator()))


@pytest.mark.parametrize("workers", [0, 2])
def test_continuous_loader_crosses_many_single_batch_passes(workers):
    data = TensorDataset(torch.arange(5))
    loader = DataLoader(data, batch_sampler=shuffled_training_batches(
        5, 4, torch.Generator().manual_seed(42)),
        num_workers=workers, persistent_workers=workers > 0)
    iterator = iter(loader)
    processes = tuple(iterator._workers) if workers else ()
    for (batch,) in itertools.islice(iterator, 12):
        assert len(batch) == 4
        assert len(set(batch.tolist())) == 4
    if workers:
        assert tuple(iterator._workers) == processes
        assert all(process.is_alive() for process in processes)
        iterator._shutdown_workers()

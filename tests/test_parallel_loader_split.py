"""Validate parallel dataloader split helpers on CPU-safe toy datasets."""

from __future__ import annotations

from training.train_parallel.data_parallel import build_parallel_train_eval_loaders


class TinyDataset:
    def __init__(self, size: int):
        self.items = list(range(size))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def test_build_parallel_train_eval_loaders_splits_dataset_without_context():
    split = build_parallel_train_eval_loaders(
        TinyDataset(4),
        batch_size=2,
        shuffle=False,
        context=None,
        collate_fn=lambda batch: batch,
        eval_size=1,
        eval_shuffle=False,
        split_seed=42,
        shuffle_before_split=False,
    )

    train_loader, eval_loader, train_indices, eval_indices = split
    assert train_indices == (0, 1, 2)
    assert eval_indices == (3,)
    assert next(iter(train_loader)) == [0, 1]
    assert next(iter(eval_loader)) == [3]
"""Lightweight dataset loaders for VAE experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATA_ROOT = Path(__file__).resolve().parents[3] / "data"

DatasetName = Literal["mnist", "fashion_mnist", "cifar10"]

_DATASET_MAP: dict[str, type] = {
    "mnist": datasets.MNIST,
    "fashion_mnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
}

_NUM_CLASSES: dict[str, int] = {
    "mnist": 10,
    "fashion_mnist": 10,
    "cifar10": 10,
}


def available_datasets() -> list[str]:
    """Return names of supported datasets."""
    return list(_DATASET_MAP.keys())


def _get_transform(name: str, image_size: int = 32) -> transforms.Compose:
    """Build a transform pipeline for *name*."""
    if name in ("mnist", "fashion_mnist"):
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),  # -> [0, 1]
        ])
    else:  # cifar10
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ])


def get_dataloader(
    name: DatasetName = "mnist",
    *,
    train: bool = True,
    batch_size: int = 128,
    image_size: int = 32,
    num_workers: int = 0,  # 0 is safest on macOS / MPS
    pin_memory: bool = False,
    data_root: str | Path | None = None,
) -> DataLoader:
    """Return a ``DataLoader`` for the given dataset.

    Parameters
    ----------
    name : str
        One of ``"mnist"``, ``"fashion_mnist"``, ``"cifar10"``.
    train : bool
        ``True`` for training set, ``False`` for test set.
    batch_size : int
        Mini-batch size.
    image_size : int
        Images are resized to ``(image_size, image_size)``.
    """
    if name not in _DATASET_MAP:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {available_datasets()}")

    root = Path(data_root) if data_root else DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)

    transform = _get_transform(name, image_size=image_size)
    ds = _DATASET_MAP[name](
        root=str(root),
        train=train,
        download=True,
        transform=transform,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=train,
    )


def num_classes(name: DatasetName) -> int:
    """Return number of classes for a dataset."""
    return _NUM_CLASSES[name]


def in_channels(name: DatasetName) -> int:
    """Return the number of image channels for *name*."""
    return 1 if name in ("mnist", "fashion_mnist") else 3

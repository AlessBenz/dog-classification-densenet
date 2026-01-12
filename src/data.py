from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from tqdm import tqdm


def compute_mean_std(train_dir: str | Path, batch_size: int = 64, num_workers: int = 2) -> Tuple[list[float], list[float]]:
    train_dir = Path(train_dir)
    ds = datasets.ImageFolder(
        root=str(train_dir),
        transform=transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()]),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    mean = torch.zeros(3)
    var = torch.zeros(3)
    n = 0

    for x, _ in tqdm(loader, desc="Computing mean/std", leave=False):
        b = x.size(0)
        x = x.view(b, 3, -1)
        mean += x.mean(dim=2).sum(dim=0)
        var += x.var(dim=2, unbiased=False).sum(dim=0)
        n += b

    mean /= n
    var /= n
    std = torch.sqrt(var)
    return mean.tolist(), std.tolist()


def build_transforms(mean: list[float], std: list[float]):
    train_tfms = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_tfms, eval_tfms


def make_loaders(
    data_dir: str | Path,
    mean: list[float],
    std: list[float],
    batch_size: int,
    num_workers: int = 2,
    balanced: bool = False,
):
    data_dir = Path(data_dir)
    train_tfms, eval_tfms = build_transforms(mean, std)

    train_ds = datasets.ImageFolder(root=str(data_dir / "train"), transform=train_tfms)
    val_ds = datasets.ImageFolder(root=str(data_dir / "val"), transform=eval_tfms)
    test_ds = datasets.ImageFolder(root=str(data_dir / "test"), transform=eval_tfms)

    sampler = None
    if balanced:
        targets = torch.tensor(train_ds.targets)
        class_counts = torch.bincount(targets)
        class_weights = 1.0 / torch.clamp(class_counts.float(), min=1.0)
        sample_weights = class_weights[targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=sampler,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@dataclass
class Checkpoint:
    epoch: int
    best_val_acc: float
    model_state: Dict
    optimizer_state: Dict
    scaler_state: Optional[Dict] = None


def save_checkpoint(path: str | Path, ckpt: Checkpoint) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": ckpt.epoch,
            "best_val_acc": ckpt.best_val_acc,
            "model_state": ckpt.model_state,
            "optimizer_state": ckpt.optimizer_state,
            "scaler_state": ckpt.scaler_state,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict:
    return torch.load(path, map_location=device)

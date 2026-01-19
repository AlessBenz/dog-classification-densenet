from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
import yaml

from src.data import compute_mean_std, make_loaders
from src.models.densenet import build_densenet121, build_from_scratch
from src.utils import load_checkpoint, save_json


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)

        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--use_pretrained", action="store_true", help="Use ImageNet weights (transfer learning).")
    ap.add_argument("--fine_tune", type=str, default="none", choices=["none", "last_block", "all"])
    ap.add_argument("--out_json", type=str, default="eval_metrics.json")
    ap.add_argument("--no_test", action="store_true", help="Skip test evaluation (if no test split).")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    data_dir = args.data_dir or cfg["data"]["data_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # mean/std
    mean = cfg["data"].get("mean")
    std = cfg["data"].get("std")
    if mean is None or std is None:
        mean, std = compute_mean_std(Path(data_dir) / "train")

    # loaders and classes
    has_test = (not args.no_test) and bool(cfg.get("data", {}).get("has_test", True))
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = make_loaders(
        data_dir=data_dir,
        mean=mean,
        std=std,
        batch_size=int(cfg["train"]["batch_size"]),
        num_workers=int(cfg["data"].get("num_workers", 2)),
        has_test=has_test,
    )
    num_classes = len(train_ds.classes)

    pretrained = bool(cfg["model"].get("pretrained", False)) or bool(args.use_pretrained)
    if not pretrained:
        model = build_from_scratch(
            num_classes=num_classes,
            dropout=float(cfg["model"].get("dropout", 0.3)),
            hidden_dim=int(cfg["model"].get("hidden_dim", 512)),
        )
    else:
        model = build_densenet121(
            num_classes=num_classes,
            pretrained=True,
            fine_tune=args.fine_tune,
            dropout=float(cfg["model"].get("dropout", 0.3)),
            hidden_dim=int(cfg["model"].get("hidden_dim", 512)),
        )
    model.to(device)

    ckpt = load_checkpoint(args.checkpoint, device)
    model.load_state_dict(ckpt["model_state"])

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)

    out = {"val_loss": val_loss, "val_acc": val_acc}
    if test_loader is not None:
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        out.update({"test_loss": test_loss, "test_acc": test_acc})
    save_json(args.out_json, out)
    print(out)


if __name__ == "__main__":
    main()

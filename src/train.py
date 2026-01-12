from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import yaml

from src.data import compute_mean_std, make_loaders
from src.models.densenet import build_densenet121, build_from_scratch
from src.utils import Checkpoint, accuracy_top1, ensure_dir, load_checkpoint, save_checkpoint, save_json, seed_everything


def freeze_first_n_feature_children(model: torch.nn.Module, n: int) -> None:
    if n <= 0 or not hasattr(model, "features"):
        return
    children = list(model.features.children())
    for i, child in enumerate(children):
        req = False if i < n else True
        for p in child.parameters():
            p.requires_grad = req


def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None, amp: bool = True):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if amp and device.type == "cuda":
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits.detach(), y) * bs
        n += bs
        pbar.set_postfix(loss=total_loss / max(n, 1), acc=total_acc / max(n, 1))

    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n = 0

    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy_top1(logits, y) * bs
        n += bs

    return total_loss / n, total_acc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/train.yaml")
    ap.add_argument("--data_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--resume", type=str, default=None)

    ap.add_argument("--balanced_sampler", action="store_true")
    ap.add_argument("--phase1_epochs", type=int, default=None)
    ap.add_argument("--phase2_epochs", type=int, default=0)
    ap.add_argument("--freeze_first_n", type=int, default=0)
    ap.add_argument("--phase2_lr", type=float, default=None)

    ap.add_argument("--use_pretrained", action="store_true")  # optional comparison (not for course)
    ap.add_argument("--fine_tune", type=str, default=None, choices=["none", "last_block", "all"])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed_everything(int(cfg.get("seed", 42)))

    data_dir = args.data_dir or cfg["data"]["data_dir"]
    out_dir = args.out_dir or cfg["out"]["out_dir"]
    epochs = int(args.epochs or cfg["train"]["epochs"])
    batch_size = int(args.batch_size or cfg["train"]["batch_size"])
    lr = float(args.lr or cfg["train"]["lr"])
    num_workers = int(cfg["data"].get("num_workers", 2))

    pretrained = bool(cfg["model"].get("pretrained", False)) or bool(args.use_pretrained)
    fine_tune = args.fine_tune or cfg["model"].get("fine_tune", "none")
    dropout = float(cfg["model"].get("dropout", 0.3))
    hidden_dim = int(cfg["model"].get("hidden_dim", 512))

    amp = bool(cfg["train"].get("amp", True))
    patience = cfg["train"].get("patience", 20)
    if patience is not None:
        patience = int(patience)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = ensure_dir(out_dir)
    ckpt_dir = ensure_dir(run_dir / "checkpoints")

    mean = cfg["data"].get("mean")
    std = cfg["data"].get("std")
    if mean is None or std is None:
        mean, std = compute_mean_std(Path(data_dir) / "train", batch_size=batch_size, num_workers=num_workers)
    save_json(run_dir / "data_stats.json", {"mean": mean, "std": std})

    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = make_loaders(
        data_dir=data_dir, mean=mean, std=std, batch_size=batch_size, num_workers=num_workers, balanced=args.balanced_sampler
    )
    num_classes = len(train_ds.classes)

    if not pretrained:
        model = build_from_scratch(num_classes=num_classes, dropout=dropout, hidden_dim=hidden_dim)
    else:
        model = build_densenet121(num_classes=num_classes, pretrained=True, fine_tune=fine_tune, dropout=dropout, hidden_dim=hidden_dim)

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
    criterion = nn.CrossEntropyLoss(label_smoothing=float(cfg["train"].get("label_smoothing", 0.0)))
    scaler = GradScaler(enabled=(amp and device.type == "cuda"))

    start_epoch = 0
    best_val_acc = 0.0
    if args.resume:
        ckpt = load_checkpoint(args.resume, device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        if ckpt.get("scaler_state"):
            scaler.load_state_dict(ckpt["scaler_state"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_acc = float(ckpt.get("best_val_acc", 0.0))

    history = []
    bad_epochs = 0

    phase1_epochs = args.phase1_epochs if args.phase1_epochs is not None else epochs
    phase1_epochs = min(int(phase1_epochs), int(epochs))
    phase2_epochs = int(args.phase2_epochs or 0)

    def run_epochs(from_epoch: int, to_epoch: int):
        nonlocal best_val_acc, bad_epochs, history, optimizer
        for epoch in range(from_epoch, to_epoch):
            train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler=scaler, amp=amp)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)

            history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc, "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]})
            save_json(run_dir / "history.json", history)

            save_checkpoint(ckpt_dir / "last.pt", Checkpoint(epoch=epoch, best_val_acc=best_val_acc, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), scaler_state=scaler.state_dict()))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_checkpoint(ckpt_dir / "best.pt", Checkpoint(epoch=epoch, best_val_acc=best_val_acc, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), scaler_state=scaler.state_dict()))
                bad_epochs = 0
            else:
                bad_epochs += 1

            print(f"[epoch {epoch+1}/{epochs}] train acc={train_acc:.4f} val acc={val_acc:.4f} best={best_val_acc:.4f}")

            if patience is not None and bad_epochs >= patience:
                print(f"Early stopping: no improvement for {patience} epochs.")
                return False
        return True

    ok = run_epochs(start_epoch, phase1_epochs)

    if ok and phase2_epochs > 0 and args.freeze_first_n > 0:
        print(f"\\n=== PHASE 2: Freeze first {args.freeze_first_n} feature children for {phase2_epochs} epochs ===")
        freeze_first_n_feature_children(model, args.freeze_first_n)

        phase2_lr = float(args.phase2_lr) if args.phase2_lr is not None else optimizer.param_groups[0]["lr"] * 0.3
        params2 = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params2, lr=phase2_lr, weight_decay=float(cfg["train"].get("weight_decay", 0.0)))
        bad_epochs = 0
        run_epochs(phase1_epochs, min(epochs, phase1_epochs + phase2_epochs))

    best_ckpt = load_checkpoint(ckpt_dir / "best.pt", device)
    model.load_state_dict(best_ckpt["model_state"])
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    save_json(run_dir / "final_metrics.json", {"best_val_acc": best_val_acc, "test_loss": test_loss, "test_acc": test_acc})
    print(f"TEST acc={test_acc:.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


def list_images(class_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]


def copy_files(files: list[Path], dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in files:
        shutil.copy2(src, dst_dir / src.name)


def main():
    ap = argparse.ArgumentParser(
        description="Split an ImageFolder dataset into train/val[/test] with a deterministic per-class split."
    )
    ap.add_argument("--images_dir", type=str, required=True, help="Path to Images/ directory containing class folders.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for split (train/val[/test]).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val", type=float, default=0.2)
    ap.add_argument("--test", type=float, default=0.0, help="Set to 0 to create no test split.")
    ap.add_argument("--overwrite", action="store_true", help="If out_dir exists, delete it first.")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)

    if not images_dir.exists():
        raise FileNotFoundError(images_dir)

    ratios = (float(args.train), float(args.val), float(args.test))
    if any(r < 0 for r in ratios):
        raise ValueError("train/val/test ratios must be >= 0")
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")

    if out_dir.exists() and args.overwrite:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(int(args.seed))

    classes = sorted([p for p in images_dir.iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found in {images_dir}")

    for class_path in classes:
        imgs = list_images(class_path)
        if not imgs:
            continue
        rng.shuffle(imgs)

        n = len(imgs)
        n_train = int(round(n * float(args.train)))
        n_val = int(round(n * float(args.val)))
        # ensure sums to n
        n_train = min(max(n_train, 1), n)  # keep at least 1 train if possible
        n_val = min(max(n_val, 0), n - n_train)
        n_test = n - n_train - n_val

        train_files = imgs[:n_train]
        val_files = imgs[n_train : n_train + n_val]
        test_files = imgs[n_train + n_val :]

        copy_files(train_files, out_dir / "train" / class_path.name)
        copy_files(val_files, out_dir / "val" / class_path.name)
        if n_test > 0:
            copy_files(test_files, out_dir / "test" / class_path.name)

    print(f"Done. Created split at: {out_dir}")
    if float(args.test) == 0.0:
        print("(No test split created.)")


if __name__ == "__main__":
    main()

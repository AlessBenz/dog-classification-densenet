from __future__ import annotations

import argparse
from pathlib import Path

import splitfolders


def main():
    ap = argparse.ArgumentParser(description="Split an ImageFolder dataset into train/val/test.")
    ap.add_argument("--images_dir", type=str, required=True, help="Path to Images/ directory containing class folders.")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory for split (train/val/test).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train", type=float, default=0.7)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    args = ap.parse_args()

    if abs((args.train + args.val + args.test) - 1.0) > 1e-6:
        raise ValueError("train+val+test must sum to 1.0")

    splitfolders.ratio(
        str(Path(args.images_dir)),
        output=str(Path(args.out_dir)),
        seed=args.seed,
        ratio=(args.train, args.val, args.test),
        group_prefix=None,
        move=False,
    )
    print(f"Done. Created split at: {args.out_dir}")


if __name__ == "__main__":
    main()

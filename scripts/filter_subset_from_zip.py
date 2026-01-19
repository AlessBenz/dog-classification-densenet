from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import zipfile


def parse_class_name(class_folder: str) -> str | None:
    """
    Kaggle Stanford Dogs zip usually contains folders like:
      images/Images/n02085620-Chihuahua/xxx.jpg
    This returns "Chihuahua" from "n02085620-Chihuahua".
    """
    if "-" not in class_folder:
        return None
    return class_folder.split("-", 1)[1]


def main():
    ap = argparse.ArgumentParser(
        description="Create a curated subset from the Stanford Dogs ZIP by filtering breeds by image count threshold."
    )
    ap.add_argument("--zip_path", type=str, required=True, help="Path to the Stanford Dogs dataset zip (Kaggle download).")
    ap.add_argument("--out_dir", type=str, default="filtered_dataset", help="Output directory to write the filtered subset.")
    ap.add_argument("--threshold", type=int, default=195, help="Keep breeds with >= threshold images.")
    ap.add_argument("--top_n", type=int, default=None, help="Alternative to threshold: keep the top-N breeds by image count.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true", help="Only print selected breeds, do not extract files.")
    ap.add_argument("--make_zip", action="store_true", help="Also create a zip archive of out_dir.")
    args = ap.parse_args()

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    out_dir = Path(args.out_dir)
    images_prefix = "images/Images/"

    # 1) Count images per class (mirrors the notebook logic)
    class_counts: dict[str, int] = {}
    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.startswith(images_prefix) and name.lower().endswith(".jpg"):
                parts = name.split("/")
                if len(parts) < 3:
                    continue
                class_folder = parts[2]
                class_name = parse_class_name(class_folder)
                if class_name is None:
                    continue
                class_counts[class_name] = class_counts.get(class_name, 0) + 1

    if not class_counts:
        raise RuntimeError("No images found. Check that the zip has paths like images/Images/<class>/...jpg")

    # 2) Select classes
    items = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)

    if args.top_n is not None:
        selected = dict(items[: int(args.top_n)])
    else:
        selected = {k: v for k, v in items if v >= int(args.threshold)}

    selected_names = list(selected.keys())
    random.Random(args.seed).shuffle(selected_names)

    print(f"Found {len(class_counts)} breeds in zip.")
    print(f"Selected {len(selected)} breeds.")
    print("Top selected breeds (name -> count):")
    for k in sorted(selected.keys(), key=lambda x: selected[x], reverse=True)[:20]:
        print(f"  {k:25s} {selected[k]}")

    manifest = {
        "zip_path": str(zip_path),
        "threshold": args.threshold,
        "top_n": args.top_n,
        "num_breeds_in_zip": len(class_counts),
        "num_selected_breeds": len(selected),
        "selected_breeds": selected,
    }

    if args.dry_run:
        return

    # 3) Extract only selected breeds (images only)
    out_images_root = out_dir / "images" / "Images"
    out_images_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        for name in z.namelist():
            if name.startswith(images_prefix) and name.lower().endswith(".jpg"):
                parts = name.split("/")
                if len(parts) < 4:
                    continue
                class_folder = parts[2]
                class_name = parse_class_name(class_folder)
                if class_name is None or class_name not in selected:
                    continue

                dest = out_dir / name  # preserves images/Images/...
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())

    (out_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nWrote subset to: {out_dir}")
    print(f"Manifest: {out_dir / 'subset_manifest.json'}")

    if args.make_zip:
        import shutil
        zip_out = out_dir.with_suffix(".zip")
        if zip_out.exists():
            zip_out.unlink()
        shutil.make_archive(str(out_dir), "zip", str(out_dir))
        print(f"Created archive: {zip_out}")


if __name__ == "__main__":
    main()

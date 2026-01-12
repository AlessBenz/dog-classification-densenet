from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import zipfile


def parse_class_name(class_folder: str) -> str | None:
    if "-" not in class_folder:
        return None
    return class_folder.split("-", 1)[1]


def main():
    ap = argparse.ArgumentParser(description="Create a curated subset from the Stanford Dogs ZIP by filtering breeds by image count threshold.")
    ap.add_argument("--zip_path", type=str, required=True, help="Path to the Stanford Dogs dataset zip (Kaggle download).")
    ap.add_argument("--out_dir", type=str, default="filtered_dataset", help="Output directory to write the filtered subset.")
    ap.add_argument("--threshold", type=int, default=195, help="Keep breeds with >= threshold images.")
    ap.add_argument("--top_n", type=int, default=None, help="Alternative to threshold: keep top-N breeds by image count.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry_run", action="store_true", help="Only print selected breeds, do not extract files.")
    args = ap.parse_args()

    zip_path = Path(args.zip_path)
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    out_dir = Path(args.out_dir)
    images_prefix = "images/Images/"

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

    items = sorted(class_counts.items(), key=lambda kv: kv[1], reverse=True)

    if args.top_n is not None:
        selected = dict(items[: int(args.top_n)])
    else:
        selected = {k: v for k, v in items if v >= int(args.threshold)}

    print(f"Found {len(class_counts)} breeds in zip.")
    print(f"Selected {len(selected)} breeds.")

    if args.dry_run:
        print("Top selected breeds (name -> count):")
        for k in sorted(selected.keys(), key=lambda x: selected[x], reverse=True)[:20]:
            print(f"  {k:25s} {selected[k]}")
        return

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

                dest = out_dir / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, open(dest, "wb") as dst:
                    dst.write(src.read())

    manifest = {
        "zip_path": str(zip_path),
        "threshold": args.threshold,
        "top_n": args.top_n,
        "num_breeds_in_zip": len(class_counts),
        "num_selected_breeds": len(selected),
        "selected_breeds": selected,
    }
    (out_dir / "subset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote subset to: {out_dir}")
    print(f"Manifest: {out_dir / 'subset_manifest.json'}")


if __name__ == "__main__":
    main()

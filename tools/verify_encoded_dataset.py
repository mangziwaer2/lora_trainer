#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
LOSSY_EXTENSIONS = (".jpg", ".jpeg", ".webp")


def decode_image_simple(enc_path: Path, key: int) -> Image.Image:
    encoded_img = Image.open(enc_path).convert("RGB")
    encoded_array = np.array(encoded_img)
    np.random.seed(key)
    random_mask = np.random.randint(0, 256, encoded_array.shape, dtype=np.uint8)
    decoded = np.bitwise_xor(encoded_array, random_mask)
    return Image.fromarray(decoded.astype(np.uint8))


def find_original_path(original_dir: Path, encoded_path: Path) -> Path | None:
    same_name = original_dir / encoded_path.name
    if same_name.exists():
        return same_name

    for ext in IMAGE_EXTENSIONS:
        candidate = original_dir / f"{encoded_path.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def image_diff_stats(decoded: Image.Image, original_path: Path) -> dict:
    original = Image.open(original_path).convert("RGB")
    decoded_array = np.array(decoded)
    original_array = np.array(original)
    stats = {
        "original_path": str(original_path),
        "shape_matches": decoded_array.shape == original_array.shape,
        "exact_match": False,
        "max_abs_diff": None,
        "mean_abs_diff": None,
    }
    if decoded_array.shape != original_array.shape:
        return stats

    diff = np.abs(decoded_array.astype(np.int16) - original_array.astype(np.int16))
    stats["exact_match"] = bool(np.array_equal(decoded_array, original_array))
    stats["max_abs_diff"] = int(diff.max())
    stats["mean_abs_diff"] = float(diff.mean())
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Decode a XOR-encoded dataset with the same logic used by the dataloader and save plaintext previews."
    )
    parser.add_argument("encoded_dir", help="Folder containing encoded training images.")
    parser.add_argument("--key", type=int, default=123456789, help="XOR decode key.")
    parser.add_argument("--output-dir", default="decoded_preview", help="Where to save decoded PNG previews and summary.json.")
    parser.add_argument("--original-dir", default=None, help="Optional original-image folder for exact pixel comparison.")
    parser.add_argument("--limit", type=int, default=24, help="Maximum decoded previews to save.")
    parser.add_argument("--allow-lossy-extensions", action="store_true", help="Do not fail when encoded files use JPG/WebP extensions.")
    args = parser.parse_args()

    encoded_dir = Path(args.encoded_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    original_dir = Path(args.original_dir).expanduser().resolve() if args.original_dir else None

    if not encoded_dir.is_dir():
        raise FileNotFoundError(f"encoded_dir does not exist or is not a directory: {encoded_dir}")
    if original_dir is not None and not original_dir.is_dir():
        raise FileNotFoundError(f"original_dir does not exist or is not a directory: {original_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in encoded_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    lossy_files = [path for path in files if path.suffix.lower() in LOSSY_EXTENSIONS]

    summary = {
        "encoded_dir": str(encoded_dir),
        "output_dir": str(output_dir),
        "key": args.key,
        "image_count": len(files),
        "lossy_encoded_count": len(lossy_files),
        "lossy_encoded_examples": [str(path) for path in lossy_files[:20]],
        "decoded": [],
    }

    compare_failures = 0
    for index, path in enumerate(files[: max(args.limit, 0)]):
        decoded = decode_image_simple(path, args.key)
        preview_path = output_dir / f"{index:04d}_{path.stem}.png"
        decoded.save(preview_path)
        item = {
            "encoded_path": str(path),
            "decoded_preview_path": str(preview_path),
            "encoded_extension": path.suffix.lower(),
        }
        if original_dir is not None:
            original_path = find_original_path(original_dir, path)
            if original_path is None:
                item["original_path"] = None
                item["compare_error"] = "matching original file not found"
                compare_failures += 1
            else:
                diff_stats = image_diff_stats(decoded, original_path)
                item.update(diff_stats)
                if not diff_stats["exact_match"]:
                    compare_failures += 1
        summary["decoded"].append(item)

    summary["compare_failures"] = compare_failures
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDecoded previews written to: {output_dir}")

    if lossy_files and not args.allow_lossy_extensions:
        print("\nERROR: encoded dataset contains JPG/JPEG/WebP files. XOR-encoded images must be saved losslessly as PNG.")
        return 2
    if original_dir is not None and compare_failures:
        print("\nERROR: at least one decoded image did not exactly match its original.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

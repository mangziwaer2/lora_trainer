#!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent
SD_SCRIPTS_ROOT = REPO_ROOT / "vendor" / "sd-scripts"
if str(SD_SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SD_SCRIPTS_ROOT))

from library.train_util import BucketManager, load_image, trim_and_resize_if_required  # noqa: E402


def encode_image_simple(img_path: Path, output_path: Path, key: int = 123456789) -> Path:
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img)
    np.random.seed(key)
    random_mask = np.random.randint(0, 256, img_array.shape, dtype=np.uint8)
    encoded = np.bitwise_xor(img_array, random_mask)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(encoded.astype(np.uint8)).save(output_path)
    return output_path


def create_demo_image(output_path: Path, width: int = 1024, height: int = 1536) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
    r = np.broadcast_to(x, (height, width))
    g = np.broadcast_to(y, (height, width))
    b = ((r.astype(np.uint16) + g.astype(np.uint16)) // 2).astype(np.uint8)
    img = np.stack([r, g, b], axis=-1)
    Image.fromarray(img, "RGB").save(output_path)
    return output_path


def image_diff_stats(a: np.ndarray, b: np.ndarray) -> dict:
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    return {
        "shape_matches": a.shape == b.shape,
        "exact_match": bool(np.array_equal(a, b)),
        "max_abs_diff": int(diff.max()) if a.shape == b.shape else None,
        "mean_abs_diff": float(diff.mean()) if a.shape == b.shape else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that Anima's current training-time decode path restores an encoded image correctly."
    )
    parser.add_argument("--image", default=None, help="Optional plaintext source image. If omitted, a synthetic demo image is generated.")
    parser.add_argument("--workdir", default=str(REPO_ROOT / "temp" / "decode_pipeline_check"), help="Directory to write artifacts.")
    parser.add_argument("--key", type=int, default=123456789, help="XOR encode/decode key.")
    parser.add_argument("--resolution", type=int, default=1024, help="Base training resolution used by bucket selection.")
    parser.add_argument("--min-bucket-reso", type=int, default=512)
    parser.add_argument("--max-bucket-reso", type=int, default=1664)
    parser.add_argument("--bucket-reso-steps", type=int, default=64)
    parser.add_argument("--bucket-no-upscale", action="store_true")
    args = parser.parse_args()

    workdir = Path(args.workdir).expanduser().resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    if args.image:
        original_path = Path(args.image).expanduser().resolve()
        if not original_path.exists():
            raise FileNotFoundError(f"Original image does not exist: {original_path}")
    else:
        original_path = create_demo_image(workdir / "original_demo.png")

    encoded_path = workdir / f"{original_path.stem}_encoded.png"
    encode_image_simple(original_path, encoded_path, key=args.key)

    original = np.array(Image.open(original_path).convert("RGB"))
    decoded_by_loader = load_image(str(encoded_path), decode_images=True, decode_key=args.key)
    decoded_path = workdir / f"{original_path.stem}_decoded_by_loader.png"
    Image.fromarray(decoded_by_loader).save(decoded_path)

    manager = BucketManager(
        args.bucket_no_upscale,
        (args.resolution, args.resolution),
        args.min_bucket_reso,
        args.max_bucket_reso,
        args.bucket_reso_steps,
    )
    if not args.bucket_no_upscale:
        manager.make_buckets()
    bucket_reso, resized_size, ar_error = manager.select_bucket(original.shape[1], original.shape[0])
    processed, _, _ = trim_and_resize_if_required(False, decoded_by_loader, bucket_reso, resized_size)
    processed_path = workdir / f"{original_path.stem}_train_input.png"
    Image.fromarray(processed).save(processed_path)

    summary = {
        "original_path": str(original_path),
        "encoded_path": str(encoded_path),
        "decoded_by_loader_path": str(decoded_path),
        "train_input_path": str(processed_path),
        "decode_key": args.key,
        "bucket_resolution": list(bucket_reso),
        "resized_size": list(resized_size),
        "aspect_ratio_error": float(ar_error),
        "decode_vs_original": image_diff_stats(decoded_by_loader, original),
    }

    summary_path = workdir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

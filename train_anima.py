#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import toml
import yaml
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
LOSSY_EXTENSIONS = {".jpg", ".jpeg", ".webp"}


DEFAULTS: dict[str, Any] = {
    "caption_ext": "txt",
    "decode_images": True,
    "decode_key": 123456789,
    "allow_lossy_extensions": False,
    "decoded_dataset_dir": None,
    "write_config_only": False,
    "keep_decoded_dataset": True,
    "sd_scripts_python": None,
    "num_cpu_threads_per_process": 1,
    "batch_size": 1,
    "num_repeats": 1,
    "resolution": 1024,
    "enable_bucket": True,
    "bucket_reso_steps": 64,
    "min_bucket_reso": 512,
    "max_bucket_reso": 1536,
    "bucket_no_upscale": False,
    "flip_aug": False,
    "shuffle_caption": False,
    "seed": 42,
    "network_module": "networks.lora_anima",
    "network_dim": 8,
    "network_alpha": None,
    "learning_rate": 1e-4,
    "optimizer_type": "AdamW8bit",
    "lr_scheduler": "constant",
    "timestep_sampling": "sigmoid",
    "discrete_flow_shift": 1.0,
    "max_train_epochs": 10,
    "max_train_steps": None,
    "save_every_n_epochs": 1,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "cache_latents": True,
    "cache_text_encoder_outputs": True,
    "vae_chunk_size": 64,
    "vae_disable_cache": True,
    "save_model_as": "safetensors",
    "llm_adapter_path": None,
    "t5_tokenizer_path": None,
    "extra_args": [],
    "command_path": None,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Anima LoRA training using sd-scripts, with optional decoding for XOR-encoded datasets.",
    )
    parser.add_argument(
        "--config-file",
        default=r"E:\project\loraTrainer\compressed\ai-toolkit\config\anima_lora_cli.example.yaml",
        help="Path to a simple YAML or JSON config file.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset folder path.")
    parser.add_argument("--output", default=None, help="Output root folder.")
    parser.add_argument("--name", default=None, help="Training job name.")
    parser.add_argument("--sd-scripts-root", default=None, help="Path to a local sd-scripts checkout.")
    parser.add_argument("--sd-scripts-python", default=None, help="Python executable for the sd-scripts environment.")
    parser.add_argument("--anima-model", default=None, help="Path to the Anima DiT .safetensors file.")
    parser.add_argument("--qwen3", default=None, help="Path to the Qwen3-0.6B directory or .safetensors file.")
    parser.add_argument("--vae", default=None, help="Path to the Qwen-Image VAE file.")
    parser.add_argument("--llm-adapter-path", default=None, help="Optional path to a separate LLM adapter file.")
    parser.add_argument("--t5-tokenizer-path", default=None, help="Optional path to a T5 tokenizer directory.")
    parser.add_argument("--caption-ext", default=None, help="Caption file extension without dot, for example txt.")
    parser.add_argument(
        "--decode-images",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Decode XOR-encoded dataset images before training.",
    )
    parser.add_argument("--decode-key", type=int, default=None, help="Decode key for XOR-encoded image datasets.")
    parser.add_argument(
        "--allow-lossy-extensions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Allow JPG/JPEG/WebP input files when decoding encoded datasets.",
    )
    parser.add_argument("--decoded-dataset-dir", default=None, help="Optional explicit decoded dataset output directory.")
    parser.add_argument(
        "--keep-decoded-dataset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep the decoded dataset on disk after the run.",
    )
    parser.add_argument("--num-cpu-threads-per-process", type=int, default=None, help="accelerate launch CPU thread count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-step batch size.")
    parser.add_argument("--num-repeats", type=int, default=None, help="Dataset subset repeat count.")
    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=None,
        help="Resolution as one integer or width height, for example 1024 or 1024 1024.",
    )
    parser.add_argument(
        "--enable-bucket",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable aspect-ratio bucketing in sd-scripts.",
    )
    parser.add_argument("--bucket-reso-steps", type=int, default=None, help="Bucket resolution step size.")
    parser.add_argument("--min-bucket-reso", type=int, default=None, help="Minimum bucket resolution.")
    parser.add_argument("--max-bucket-reso", type=int, default=None, help="Maximum bucket resolution.")
    parser.add_argument(
        "--bucket-no-upscale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable bucket upscaling for undersized images.",
    )
    parser.add_argument("--flip-aug", action=argparse.BooleanOptionalAction, default=None, help="Enable horizontal flip augmentation.")
    parser.add_argument(
        "--shuffle-caption",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Shuffle comma-separated caption tags in sd-scripts.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--network-module", default=None, help="Network module to train, usually networks.lora_anima.")
    parser.add_argument("--network-dim", type=int, default=None, help="LoRA rank.")
    parser.add_argument("--network-alpha", type=int, default=None, help="Optional LoRA alpha.")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--optimizer-type", default=None, help="Optimizer type.")
    parser.add_argument("--lr-scheduler", default=None, help="Learning-rate scheduler.")
    parser.add_argument("--timestep-sampling", default=None, help="Anima timestep sampling mode.")
    parser.add_argument("--discrete-flow-shift", type=float, default=None, help="Discrete flow shift.")
    parser.add_argument("--max-train-epochs", type=int, default=None, help="Maximum training epochs.")
    parser.add_argument("--max-train-steps", type=int, default=None, help="Optional explicit max training steps.")
    parser.add_argument("--save-every-n-epochs", type=int, default=None, help="Checkpoint save interval in epochs.")
    parser.add_argument("--mixed-precision", default=None, help="Mixed precision mode, for example bf16.")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--cache-latents",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache latents in sd-scripts.",
    )
    parser.add_argument(
        "--cache-text-encoder-outputs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache frozen text encoder outputs in sd-scripts.",
    )
    parser.add_argument("--vae-chunk-size", type=int, default=None, help="VAE chunk size.")
    parser.add_argument(
        "--vae-disable-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable the VAE internal cache.",
    )
    parser.add_argument("--save-model-as", default=None, choices=["safetensors", "ckpt", "pt"], help="Output format.")
    parser.add_argument("--command-path", default=None, help="Optional path to write the generated launch command JSON.")
    parser.add_argument("--write-config-only", action="store_true", default=None, help="Only write generated config files and exit.")
    return parser.parse_args()


def load_simple_config(config_file: str) -> dict[str, Any]:
    path = Path(config_file).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_file}")

    raw_text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        loaded = yaml.safe_load(raw_text)
    else:
        loaded = json.loads(raw_text)

    if not isinstance(loaded, dict):
        raise ValueError("Config file must contain a top-level mapping/object.")
    return loaded


def merge_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = dict(DEFAULTS)
    if args.config_file:
        settings.update(load_simple_config(args.config_file))

    for key, value in vars(args).items():
        if key == "config_file":
            continue
        if value is not None:
            settings[key] = value

    required_fields = ["dataset", "output", "name", "sd_scripts_root", "anima_model", "qwen3", "vae"]
    missing = [field for field in required_fields if not settings.get(field)]
    if missing:
        raise ValueError(f"Missing required settings: {', '.join(missing)}")

    resolution = settings["resolution"]
    if isinstance(resolution, int):
        pass
    elif isinstance(resolution, list) and len(resolution) in {1, 2} and all(isinstance(item, int) for item in resolution):
        settings["resolution"] = resolution[0] if len(resolution) == 1 else resolution
    else:
        raise ValueError("resolution must be an int or a 1-2 item int list, for example 1024 or [1024, 1024]")

    if settings["batch_size"] < 1:
        raise ValueError("batch_size must be >= 1")
    if settings["num_repeats"] < 1:
        raise ValueError("num_repeats must be >= 1")
    if settings["bucket_reso_steps"] < 1:
        raise ValueError("bucket_reso_steps must be >= 1")
    if settings["min_bucket_reso"] < 1 or settings["max_bucket_reso"] < settings["min_bucket_reso"]:
        raise ValueError("Bucket resolution range is invalid")
    if settings["network_dim"] < 1:
        raise ValueError("network_dim must be >= 1")
    if settings["max_train_epochs"] < 1:
        raise ValueError("max_train_epochs must be >= 1")
    if settings["max_train_steps"] is not None and settings["max_train_steps"] < 1:
        raise ValueError("max_train_steps must be >= 1 when provided")
    if settings["num_cpu_threads_per_process"] < 1:
        raise ValueError("num_cpu_threads_per_process must be >= 1")
    if not isinstance(settings["extra_args"], list):
        raise ValueError("extra_args must be a list of strings")

    return settings


def normalize_existing_path(raw_path: str, label: str) -> str:
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {raw_path}")
    return str(path.resolve())


def normalize_optional_path(raw_path: str | None, label: str) -> str | None:
    if raw_path is None:
        return None
    return normalize_existing_path(raw_path, label)


def decode_image_simple(enc_path: Path, key: int) -> Image.Image:
    encoded_img = Image.open(enc_path).convert("RGB")
    encoded_array = np.array(encoded_img)
    np.random.seed(key)
    random_mask = np.random.randint(0, 256, encoded_array.shape, dtype=np.uint8)
    decoded = np.bitwise_xor(encoded_array, random_mask)
    return Image.fromarray(decoded.astype(np.uint8))


def find_dataset_image_dirs(dataset_root: Path) -> list[Path]:
    image_dirs = sorted({path.parent for path in dataset_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS})
    if not image_dirs:
        raise FileNotFoundError(f"No supported image files found under dataset: {dataset_root}")
    return image_dirs


def decode_dataset(settings: dict[str, Any], job_root: Path) -> tuple[list[Path], Path]:
    source_root = Path(normalize_existing_path(settings["dataset"], "Dataset path"))
    target_root = Path(settings["decoded_dataset_dir"]).expanduser().resolve() if settings["decoded_dataset_dir"] else job_root / "decoded_dataset"
    target_root.mkdir(parents=True, exist_ok=True)

    caption_ext = str(settings["caption_ext"]).lstrip(".")
    decode_key = int(settings["decode_key"])
    allow_lossy = bool(settings["allow_lossy_extensions"])

    image_dirs = find_dataset_image_dirs(source_root)
    decoded_dirs: list[Path] = []
    seen_outputs: set[Path] = set()

    for image_dir in image_dirs:
        relative_dir = image_dir.relative_to(source_root)
        decoded_dir = target_root / relative_dir
        decoded_dir.mkdir(parents=True, exist_ok=True)
        decoded_dirs.append(decoded_dir)

        for image_path in sorted(image_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if image_path.suffix.lower() in LOSSY_EXTENSIONS and not allow_lossy:
                raise ValueError(
                    f"Encoded dataset contains lossy file {image_path}. Use PNG for XOR-encoded images or pass allow_lossy_extensions=true."
                )

            output_path = decoded_dir / f"{image_path.stem}.png"
            if output_path in seen_outputs:
                raise ValueError(f"Duplicate decoded output path detected: {output_path}")
            seen_outputs.add(output_path)

            decoded = decode_image_simple(image_path, key=decode_key)
            decoded.save(output_path)

            caption_path = image_path.with_suffix(f".{caption_ext}")
            if caption_path.exists():
                shutil.copy2(caption_path, decoded_dir / caption_path.name)

    manifest = {
        "source_dataset": str(source_root),
        "decoded_dataset": str(target_root),
        "decode_images": True,
        "decode_key": decode_key,
        "subset_dirs": [str(path) for path in decoded_dirs],
    }
    (job_root / "decoded_dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return decoded_dirs, target_root


def collect_plain_dataset(settings: dict[str, Any]) -> tuple[list[Path], Path]:
    source_root = Path(normalize_existing_path(settings["dataset"], "Dataset path"))
    return find_dataset_image_dirs(source_root), source_root


def build_dataset_config(settings: dict[str, Any], image_dirs: list[Path], output_path: Path) -> Path:
    dataset_config = {
        "general": {
            "caption_extension": f".{str(settings['caption_ext']).lstrip('.')}",
            "shuffle_caption": bool(settings["shuffle_caption"]),
        },
        "datasets": [
            {
                "resolution": settings["resolution"],
                "batch_size": int(settings["batch_size"]),
                "enable_bucket": bool(settings["enable_bucket"]),
                "bucket_reso_steps": int(settings["bucket_reso_steps"]),
                "min_bucket_reso": int(settings["min_bucket_reso"]),
                "max_bucket_reso": int(settings["max_bucket_reso"]),
                "bucket_no_upscale": bool(settings["bucket_no_upscale"]),
                "subsets": [
                    {
                        "image_dir": str(path),
                        "num_repeats": int(settings["num_repeats"]),
                        "flip_aug": bool(settings["flip_aug"]),
                    }
                    for path in image_dirs
                ],
            }
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(toml.dumps(dataset_config), encoding="utf-8")
    return output_path


def build_command(settings: dict[str, Any], dataset_config_path: Path) -> tuple[list[str], Path]:
    sd_scripts_root = Path(normalize_existing_path(settings["sd_scripts_root"], "sd-scripts root"))
    script_path = sd_scripts_root / "anima_train_network.py"
    if not script_path.exists():
        raise FileNotFoundError(f"sd-scripts does not contain anima_train_network.py: {script_path}")

    python_executable = settings["sd_scripts_python"] or sys.executable
    command = [
        python_executable,
        "-m",
        "accelerate.commands.launch",
        "--num_cpu_threads_per_process",
        str(int(settings["num_cpu_threads_per_process"])),
        str(script_path),
        "--pretrained_model_name_or_path",
        normalize_existing_path(settings["anima_model"], "Anima model"),
        "--qwen3",
        normalize_existing_path(settings["qwen3"], "Qwen3 model"),
        "--vae",
        normalize_existing_path(settings["vae"], "Qwen-Image VAE"),
        "--dataset_config",
        str(dataset_config_path),
        "--output_dir",
        str(Path(settings["output"]).expanduser().resolve()),
        "--output_name",
        settings["name"],
        "--save_model_as",
        settings["save_model_as"],
        "--network_module",
        settings["network_module"],
        "--network_dim",
        str(int(settings["network_dim"])),
        "--learning_rate",
        str(settings["learning_rate"]),
        "--optimizer_type",
        settings["optimizer_type"],
        "--lr_scheduler",
        settings["lr_scheduler"],
        "--timestep_sampling",
        settings["timestep_sampling"],
        "--discrete_flow_shift",
        str(settings["discrete_flow_shift"]),
        "--max_train_epochs",
        str(int(settings["max_train_epochs"])),
        "--save_every_n_epochs",
        str(int(settings["save_every_n_epochs"])),
        "--mixed_precision",
        settings["mixed_precision"],
        "--seed",
        str(int(settings["seed"])),
        "--vae_chunk_size",
        str(int(settings["vae_chunk_size"])),
    ]

    if settings["network_alpha"] is not None:
        command.extend(["--network_alpha", str(int(settings["network_alpha"]))])
    if settings["max_train_steps"] is not None:
        command.extend(["--max_train_steps", str(int(settings["max_train_steps"]))])
    if settings["llm_adapter_path"] is not None:
        command.extend(["--llm_adapter_path", normalize_existing_path(settings["llm_adapter_path"], "LLM adapter path")])
    if settings["t5_tokenizer_path"] is not None:
        command.extend(["--t5_tokenizer_path", normalize_existing_path(settings["t5_tokenizer_path"], "T5 tokenizer path")])
    if settings["gradient_checkpointing"]:
        command.append("--gradient_checkpointing")
    if settings["cache_latents"]:
        command.append("--cache_latents")
    if settings["cache_text_encoder_outputs"]:
        command.append("--cache_text_encoder_outputs")
    if settings["vae_disable_cache"]:
        command.append("--vae_disable_cache")
    if settings["extra_args"]:
        command.extend(str(item) for item in settings["extra_args"])

    return command, sd_scripts_root


def write_command(command: list[str], command_path: Path) -> Path:
    payload = {"command": command}
    command_path.parent.mkdir(parents=True, exist_ok=True)
    command_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return command_path


def run_training(command: list[str], workdir: Path) -> int:
    print("Launching:", " ".join(json.dumps(part) for part in command))
    result = subprocess.run(command, cwd=str(workdir))
    return result.returncode


def main() -> int:
    args = parse_args()
    settings = merge_settings(args)

    output_root = Path(settings["output"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    job_root = output_root / settings["name"]
    job_root.mkdir(parents=True, exist_ok=True)

    if settings["decode_images"]:
        image_dirs, decoded_root = decode_dataset(settings, job_root)
    else:
        image_dirs, decoded_root = collect_plain_dataset(settings)

    dataset_config_path = build_dataset_config(settings, image_dirs, job_root / "anima_dataset_config.toml")
    command, workdir = build_command(settings, dataset_config_path)

    command_path = Path(settings["command_path"]).expanduser().resolve() if settings["command_path"] else job_root / "anima_launch_command.json"
    write_command(command, command_path)

    summary = {
        "name": settings["name"],
        "dataset_root_used": str(decoded_root),
        "dataset_config_path": str(dataset_config_path),
        "command_path": str(command_path),
        "decode_images": bool(settings["decode_images"]),
    }
    (job_root / "anima_job_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dataset config written to: {dataset_config_path}")
    print(f"Launch command written to: {command_path}")

    if settings["write_config_only"]:
        return 0

    return run_training(command, workdir)


if __name__ == "__main__":
    raise SystemExit(main())

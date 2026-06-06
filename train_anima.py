#!/usr/bin/env python3
import argparse
import json
import os
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
IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").exists()
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "anima_lora_cli.example.yaml"
DEFAULT_SD_SCRIPTS_CANDIDATES = [
    REPO_ROOT / "vendor" / "sd-scripts",
    REPO_ROOT / "temp" / "sd-scripts",
]


DEFAULTS: dict[str, Any] = {
    "caption_ext": "txt",
    "decode_images": True,
    "decode_key": 123456789,
    "allow_lossy_extensions": False,
    "decoded_dataset_dir": None,
    "write_config_only": False,
    "keep_decoded_dataset": False,
    "sd_scripts_python": None,
    "num_cpu_threads_per_process": 1,
    "num_processes": 1 if IS_KAGGLE else None,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
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
    "max_train_epochs": None,
    "max_train_steps": None,
    "save_every_n_epochs": 1,
    "save_every_n_steps": None,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "cache_latents": True,
    "cache_text_encoder_outputs": True,
    "network_train_unet_only": True,
    "low_vram": False,
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
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to a simple YAML or JSON config file.",
    )
    parser.add_argument("--dataset", default=None, help="Dataset folder path.")
    parser.add_argument("--output", default=None, help="Output root folder.")
    parser.add_argument("--name", default=None, help="Training job name.")
    parser.add_argument(
        "--sd-scripts-root",
        default=None,
        help="Optional path to a local sd-scripts checkout. If omitted, this script auto-detects vendor/sd-scripts.",
    )
    parser.add_argument("--sd-scripts-python", default=None, help="Python executable for the sd-scripts environment.")
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Process count passed to accelerate launch. Defaults to 1 on Kaggle to avoid accidental multi-GPU startup.",
    )
    parser.add_argument("--anima-model", "--model", dest="anima_model", default=None, help="Path to the Anima DiT .safetensors file.")
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
    parser.add_argument(
        "--decoded-dataset-dir",
        default=None,
        help="Deprecated and ignored. Anima now decodes encoded images on the fly without writing a decoded dataset to disk.",
    )
    parser.add_argument(
        "--keep-decoded-dataset",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Deprecated and ignored. Anima now decodes encoded images on the fly without writing a decoded dataset to disk.",
    )
    parser.add_argument("--num-cpu-threads-per-process", type=int, default=None, help="accelerate launch CPU thread count.")
    parser.add_argument("--batch-size", type=int, default=None, help="Per-step batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        "--gradient-accumulation",
        type=int,
        dest="gradient_accumulation_steps",
        default=None,
        help="Gradient accumulation steps. This changes the effective batch size without increasing per-step VRAM.",
    )
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
    parser.add_argument("--network-dim", "--rank", dest="network_dim", type=int, default=None, help="LoRA rank.")
    parser.add_argument("--network-alpha", type=int, default=None, help="Optional LoRA alpha.")
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--optimizer-type", "--optimizer", dest="optimizer_type", default=None, help="Optimizer type.")
    parser.add_argument("--lr-scheduler", default=None, help="Learning-rate scheduler.")
    parser.add_argument("--timestep-sampling", default=None, help="Anima timestep sampling mode.")
    parser.add_argument("--discrete-flow-shift", type=float, default=None, help="Discrete flow shift.")
    parser.add_argument("--max-train-epochs", type=int, default=None, help="Maximum training epochs.")
    parser.add_argument("--max-train-steps", "--steps", dest="max_train_steps", type=int, default=None, help="Optional explicit max training steps.")
    parser.add_argument("--save-every-n-epochs", type=int, default=None, help="Checkpoint save interval in epochs.")
    parser.add_argument("--save-every-n-steps", "--save-every", dest="save_every_n_steps", type=int, default=None, help="Checkpoint save interval in steps.")
    parser.add_argument("--mixed-precision", "--dtype", dest="mixed_precision", default=None, help="Mixed precision mode, for example bf16.")
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
    parser.add_argument(
        "--network-train-unet-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train only the Anima DiT/UNet-side LoRA. This should stay enabled when caching text encoder outputs.",
    )
    parser.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Forward sd-scripts --lowram to reduce peak memory usage.",
    )
    parser.add_argument("--vae-chunk-size", type=int, default=None, help="VAE chunk size.")
    parser.add_argument(
        "--vae-disable-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable the VAE internal cache.",
    )
    parser.add_argument("--save-model-as", "--save-format", dest="save_model_as", default=None, choices=["safetensors", "ckpt", "pt"], help="Output format.")
    parser.add_argument("--command-path", default=None, help="Optional path to write the generated launch command JSON.")
    parser.add_argument("--write-config-only", action="store_true", default=None, help="Only write generated config files and exit.")
    return parser.parse_args()


def load_simple_config(config_file: str) -> tuple[dict[str, Any], Path]:
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
    return loaded, path


def normalize_config_aliases(config: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(config)
    aliases = {
        "model": "anima_model",
        "gradient_accumulation": "gradient_accumulation_steps",
        "rank": "network_dim",
        "lr": "learning_rate",
        "optimizer": "optimizer_type",
        "steps": "max_train_steps",
        "save_every": "save_every_n_steps",
        "save_format": "save_model_as",
        "dtype": "mixed_precision",
    }
    for alias, canonical in aliases.items():
        if alias in normalized and canonical not in normalized:
            normalized[canonical] = normalized[alias]
    return normalized


def merge_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings = dict(DEFAULTS)
    cli_overrides: set[str] = set()
    if args.config_file:
        loaded_config, config_path = load_simple_config(args.config_file)
        loaded_config = normalize_config_aliases(loaded_config)
        settings.update(loaded_config)
        settings["_config_dir"] = str(config_path.parent)

    for key, value in vars(args).items():
        if key == "config_file":
            continue
        if value is not None:
            settings[key] = value
            cli_overrides.add(key)

    settings["_cli_overrides"] = sorted(cli_overrides)

    required_fields = ["dataset", "output", "name", "anima_model", "qwen3", "vae"]
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
    if settings["gradient_accumulation_steps"] < 1:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if settings["num_repeats"] < 1:
        raise ValueError("num_repeats must be >= 1")
    if settings["bucket_reso_steps"] < 1:
        raise ValueError("bucket_reso_steps must be >= 1")
    if settings["min_bucket_reso"] < 1 or settings["max_bucket_reso"] < settings["min_bucket_reso"]:
        raise ValueError("Bucket resolution range is invalid")
    if settings["network_dim"] < 1:
        raise ValueError("network_dim must be >= 1")
    if settings["max_train_epochs"] is None and settings["max_train_steps"] is None:
        raise ValueError("One of max_train_steps/steps or max_train_epochs must be provided")
    if settings["max_train_epochs"] is not None and settings["max_train_epochs"] < 1:
        raise ValueError("max_train_epochs must be >= 1")
    if settings["max_train_steps"] is not None and settings["max_train_steps"] < 1:
        raise ValueError("max_train_steps must be >= 1 when provided")
    if settings["save_every_n_steps"] is not None and settings["save_every_n_steps"] < 1:
        raise ValueError("save_every_n_steps must be >= 1 when provided")
    if settings["num_processes"] is not None and settings["num_processes"] < 1:
        raise ValueError("num_processes must be >= 1 when provided")
    if settings["num_cpu_threads_per_process"] < 1:
        raise ValueError("num_cpu_threads_per_process must be >= 1")
    if not isinstance(settings["extra_args"], list):
        raise ValueError("extra_args must be a list of strings")

    return settings


def get_config_dir(settings: dict[str, Any]) -> Path | None:
    raw_dir = settings.get("_config_dir")
    if not raw_dir:
        return None
    return Path(raw_dir)


def get_base_dir_for_setting(settings: dict[str, Any], key: str) -> Path | None:
    cli_overrides = set(settings.get("_cli_overrides", []))
    if key in cli_overrides:
        return Path.cwd()
    return get_config_dir(settings)


def resolve_path(raw_path: str, base_dir: Path | None = None) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    return path.resolve()


def normalize_existing_path(raw_path: str, label: str, base_dir: Path | None = None) -> str:
    path = resolve_path(raw_path, base_dir=base_dir)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {raw_path}")
    return str(path)


def normalize_optional_path(raw_path: str | None, label: str, base_dir: Path | None = None) -> str | None:
    if raw_path is None:
        return None
    return normalize_existing_path(raw_path, label, base_dir=base_dir)


def resolve_sd_scripts_root(raw_path: str | None, base_dir: Path | None = None) -> Path:
    if raw_path:
        return Path(normalize_existing_path(raw_path, "sd-scripts root", base_dir=base_dir))

    for candidate in DEFAULT_SD_SCRIPTS_CANDIDATES:
        if candidate.exists():
            return candidate.resolve()

    searched = ", ".join(str(path) for path in DEFAULT_SD_SCRIPTS_CANDIDATES)
    raise FileNotFoundError(
        "sd-scripts root was not provided and no bundled checkout was found. "
        f"Searched: {searched}"
    )


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


def collect_plain_dataset(settings: dict[str, Any]) -> tuple[list[Path], Path]:
    source_root = Path(
        normalize_existing_path(
            settings["dataset"],
            "Dataset path",
            base_dir=get_base_dir_for_setting(settings, "dataset"),
        )
    )
    image_dirs = find_dataset_image_dirs(source_root)

    if settings["decode_images"]:
        allow_lossy = bool(settings["allow_lossy_extensions"])
        for image_dir in image_dirs:
            for image_path in sorted(image_dir.iterdir()):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                if image_path.suffix.lower() in LOSSY_EXTENSIONS and not allow_lossy:
                    raise ValueError(
                        f"Encoded dataset contains lossy file {image_path}. Use PNG for XOR-encoded images or pass allow_lossy_extensions=true."
                    )

    return image_dirs, source_root


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
                        "decode_images": bool(settings["decode_images"]),
                        "decode_key": int(settings["decode_key"]),
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
    sd_scripts_root = resolve_sd_scripts_root(
        settings.get("sd_scripts_root"),
        base_dir=get_base_dir_for_setting(settings, "sd_scripts_root"),
    )
    script_path = sd_scripts_root / "anima_train_network.py"
    if not script_path.exists():
        raise FileNotFoundError(f"sd-scripts does not contain anima_train_network.py: {script_path}")

    python_executable = (
        normalize_existing_path(
            settings["sd_scripts_python"],
            "sd-scripts Python",
            base_dir=get_base_dir_for_setting(settings, "sd_scripts_python"),
        )
        if settings["sd_scripts_python"]
        else sys.executable
    )
    command = [
        python_executable,
        "-m",
        "accelerate.commands.launch",
    ]
    if settings["num_processes"] is not None:
        command.extend(["--num_processes", str(int(settings["num_processes"]))])
    command.extend(
        [
            "--num_cpu_threads_per_process",
            str(int(settings["num_cpu_threads_per_process"])),
            str(script_path),
        "--pretrained_model_name_or_path",
        normalize_existing_path(
            settings["anima_model"],
            "Anima model",
            base_dir=get_base_dir_for_setting(settings, "anima_model"),
        ),
        "--qwen3",
        normalize_existing_path(
            settings["qwen3"],
            "Qwen3 model",
            base_dir=get_base_dir_for_setting(settings, "qwen3"),
        ),
        "--vae",
        normalize_existing_path(
            settings["vae"],
            "Qwen-Image VAE",
            base_dir=get_base_dir_for_setting(settings, "vae"),
        ),
        "--dataset_config",
        str(dataset_config_path),
        "--output_dir",
        str(resolve_path(settings["output"], base_dir=get_base_dir_for_setting(settings, "output"))),
        "--output_name",
        settings["name"],
        "--gradient_accumulation_steps",
        str(int(settings["gradient_accumulation_steps"])),
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
        "--mixed_precision",
        settings["mixed_precision"],
        "--seed",
        str(int(settings["seed"])),
        "--vae_chunk_size",
        str(int(settings["vae_chunk_size"])),
        ]
    )

    if settings["network_alpha"] is not None:
        command.extend(["--network_alpha", str(int(settings["network_alpha"]))])
    if settings["max_train_steps"] is not None:
        command.extend(["--max_train_steps", str(int(settings["max_train_steps"]))])
    elif settings["max_train_epochs"] is not None:
        command.extend(["--max_train_epochs", str(int(settings["max_train_epochs"]))])
    if settings["save_every_n_steps"] is not None:
        command.extend(["--save_every_n_steps", str(int(settings["save_every_n_steps"]))])
    elif settings["max_train_epochs"] is not None and settings["save_every_n_epochs"] is not None:
        command.extend(["--save_every_n_epochs", str(int(settings["save_every_n_epochs"]))])
    if settings["llm_adapter_path"] is not None:
        command.extend(
            [
                "--llm_adapter_path",
                normalize_existing_path(
                    settings["llm_adapter_path"],
                    "LLM adapter path",
                    base_dir=get_base_dir_for_setting(settings, "llm_adapter_path"),
                ),
            ]
        )
    if settings["t5_tokenizer_path"] is not None:
        command.extend(
            [
                "--t5_tokenizer_path",
                normalize_existing_path(
                    settings["t5_tokenizer_path"],
                    "T5 tokenizer path",
                    base_dir=get_base_dir_for_setting(settings, "t5_tokenizer_path"),
                ),
            ]
        )
    if settings["gradient_checkpointing"]:
        command.append("--gradient_checkpointing")
    if settings["cache_latents"]:
        command.append("--cache_latents")
    if settings["cache_text_encoder_outputs"]:
        command.append("--cache_text_encoder_outputs")
    if settings["network_train_unet_only"]:
        command.append("--network_train_unet_only")
    if settings["low_vram"]:
        command.append("--lowram")
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

    output_root = resolve_path(settings["output"], base_dir=get_base_dir_for_setting(settings, "output"))
    output_root.mkdir(parents=True, exist_ok=True)
    job_root = output_root / settings["name"]
    job_root.mkdir(parents=True, exist_ok=True)

    image_dirs, dataset_root = collect_plain_dataset(settings)

    dataset_config_path = build_dataset_config(settings, image_dirs, job_root / "anima_dataset_config.toml")
    command, workdir = build_command(settings, dataset_config_path)

    command_path = (
        resolve_path(settings["command_path"], base_dir=get_base_dir_for_setting(settings, "command_path"))
        if settings["command_path"]
        else job_root / "anima_launch_command.json"
    )
    write_command(command, command_path)

    summary = {
        "name": settings["name"],
        "dataset_root_used": str(dataset_root),
        "dataset_config_path": str(dataset_config_path),
        "command_path": str(command_path),
        "sd_scripts_root": str(workdir),
        "decode_images": bool(settings["decode_images"]),
        "decode_mode": "stream" if settings["decode_images"] else "plain",
        "decoded_dataset_written_to_disk": False,
        "batch_size": int(settings["batch_size"]),
        "gradient_accumulation_steps": int(settings["gradient_accumulation_steps"]),
        "effective_batch_size_single_process": int(settings["batch_size"]) * int(settings["gradient_accumulation_steps"]),
        "num_repeats": int(settings["num_repeats"]),
    }
    (job_root / "anima_job_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dataset config written to: {dataset_config_path}")
    print(f"Launch command written to: {command_path}")

    if settings["write_config_only"]:
        return 0

    return run_training(command, workdir)


if __name__ == "__main__":
    raise SystemExit(main())

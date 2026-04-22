#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").exists()


DEFAULTS: dict[str, Any] = {
    "trigger_word": None,
    "performance_log_every": 0 if IS_KAGGLE else 10,
    "disable_progress_bar": IS_KAGGLE,
    "progress_bar_mininterval": 60.0 if IS_KAGGLE else 1.0,
    "steps": 3000,
    "batch_size": 1,
    "gradient_accumulation": 1,
    "lr": 1e-4,
    "optimizer": "adamw8bit",
    "save_every": 250,
    "save_best_model": True,
    "sample_every": 250,
    "rank": 32,
    "conv_rank": 16,
    "resolution": [512, 768, 1024],
    "caption_ext": "txt",
    "cache_latents_to_disk": False,
    "train_text_encoder": False,
    "low_vram": False,
    "dtype": "fp16",
    "save_dtype": "float16",
    "save_format": "safetensors",
    "dataset_cache_dir": None,
    "sample_prompts": [],
    "sample_prompts_file": None,
    "sample_neg": "",
    "sample_width": 512,
    "sample_height": 512,
    "sample_steps": 20,
    "guidance_scale": 6.0,
    "seed": 42,
    "walk_seed": False,
    "disable_sampling": False,
    "no_skip_first_sample": False,
    "config_path": None,
    "log": None,
    "write_config_only": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch SDXL LoRA training from the command line without the UI.",
    )
    parser.add_argument("--config-file", default=None, help="Path to a simple YAML or JSON config file.")

    parser.add_argument("--dataset", default=None, help="Dataset folder path.")
    parser.add_argument(
        "--model",
        default=None,
        help="Base model path or Hugging Face model id.",
    )
    parser.add_argument("--output", default=None, help="Output root folder.")
    parser.add_argument("--name", default=None, help="Training job name.")
    parser.add_argument("--trigger-word", default=None, help="Optional trigger word.")
    parser.add_argument("--performance-log-every", type=int, default=None, help="Print timer stats every N steps. Use 0 to disable.")
    parser.add_argument(
        "--disable-progress-bar",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the tqdm training progress bar.",
    )
    parser.add_argument("--progress-bar-mininterval", type=float, default=None, help="Minimum seconds between tqdm refreshes.")

    parser.add_argument("--steps", type=int, default=None, help="Training steps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size.")
    parser.add_argument("--gradient-accumulation", type=int, default=None, help="Gradient accumulation steps.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--optimizer", default=None, help="Optimizer name.")
    parser.add_argument("--save-every", type=int, default=None, help="Checkpoint save interval.")
    parser.add_argument(
        "--save-best-model",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable auto-saving the best-loss checkpoint.",
    )
    parser.add_argument("--sample-every", type=int, default=None, help="Sample image interval.")
    parser.add_argument("--rank", type=int, default=None, help="LoRA linear rank.")
    parser.add_argument("--conv-rank", type=int, default=None, help="LoRA conv rank.")

    parser.add_argument(
        "--resolution",
        type=int,
        nargs="+",
        default=None,
        help="Bucket resolutions, for example --resolution 512 768 1024",
    )
    parser.add_argument("--caption-ext", default=None, help="Caption file extension.")
    parser.add_argument("--cache-latents-to-disk", action="store_true", default=None, help="Cache latents to disk.")
    parser.add_argument("--train-text-encoder", action="store_true", default=None, help="Enable text encoder training.")
    parser.add_argument("--low-vram", action="store_true", default=None, help="Enable low VRAM model mode.")

    parser.add_argument("--dtype", default=None, help="Training dtype, for example fp16 or bf16.")
    parser.add_argument("--save-dtype", default=None, help="Save dtype, for example float16 or bf16.")
    parser.add_argument("--save-format", default=None, choices=["safetensors", "diffusers"], help="Output format.")
    parser.add_argument("--dataset-cache-dir", default=None, help="Optional writable cache directory for dataset metadata and disk caches.")

    parser.add_argument("--sample-prompt", dest="sample_prompts", action="append", default=None, help="Repeat to add sample prompts.")
    parser.add_argument("--sample-prompts-file", default=None, help="Text file with one sample prompt per line.")
    parser.add_argument("--sample-neg", default=None, help="Negative prompt for sampling.")
    parser.add_argument("--sample-width", type=int, default=None, help="Sample image width.")
    parser.add_argument("--sample-height", type=int, default=None, help="Sample image height.")
    parser.add_argument("--sample-steps", type=int, default=None, help="Sample image inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="Guidance scale for sampling.")
    parser.add_argument("--seed", type=int, default=None, help="Base sampling seed.")
    parser.add_argument("--walk-seed", action="store_true", default=None, help="Increment seed for each sample.")
    parser.add_argument("--disable-sampling", action="store_true", default=None, help="Disable sample generation entirely.")
    parser.add_argument("--no-skip-first-sample", action="store_true", default=None, help="Generate baseline samples before training starts.")

    parser.add_argument("--config-path", default=None, help="Optional explicit generated AI Toolkit config path.")
    parser.add_argument("--log", default=None, help="Optional training log file path passed to run.py.")
    parser.add_argument("--write-config-only", action="store_true", default=None, help="Only write the config file and exit.")
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

    required_fields = ["dataset", "model", "output", "name"]
    missing = [field for field in required_fields if not settings.get(field)]
    if missing:
        raise ValueError(f"Missing required settings: {', '.join(missing)}")

    if not isinstance(settings["resolution"], list) or not settings["resolution"]:
        raise ValueError("resolution must be a non-empty list, for example [512, 768, 1024]")

    if settings.get("sample_prompts") is None:
        settings["sample_prompts"] = []

    return settings


def normalize_existing_path(raw_path: str, label: str) -> str:
    path = Path(raw_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {raw_path}")
    return str(path.resolve())


def normalize_model_path(raw_value: str) -> str:
    path = Path(raw_value).expanduser()
    if path.exists():
        return str(path.resolve())
    return raw_value


def load_sample_prompts(settings: dict[str, Any]) -> list[str]:
    prompts = list(settings.get("sample_prompts", []))
    sample_prompts_file = settings.get("sample_prompts_file")
    if sample_prompts_file:
        prompts_path = Path(sample_prompts_file).expanduser()
        if not prompts_path.exists():
            raise FileNotFoundError(f"Sample prompts file does not exist: {sample_prompts_file}")
        file_prompts = [line.strip() for line in prompts_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        prompts.extend(file_prompts)
    return prompts


def build_config(settings: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    dataset_path = normalize_existing_path(settings["dataset"], "Dataset path")
    output_root = Path(settings["output"]).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    model_path = normalize_model_path(settings["model"])
    dataset_cache_dir = settings.get("dataset_cache_dir")
    if dataset_cache_dir:
        dataset_cache_dir = Path(dataset_cache_dir).expanduser().resolve()
    else:
        dataset_cache_dir = output_root / settings["name"] / "dataset_cache"
    dataset_cache_dir.mkdir(parents=True, exist_ok=True)
    sample_prompts = load_sample_prompts(settings)

    disable_sampling = settings["disable_sampling"] or len(sample_prompts) == 0
    sample_items = [{"prompt": prompt} for prompt in sample_prompts]

    return {
        "job": "extension",
        "config": {
            "name": settings["name"],
            "process": [
                {
                    "type": "diffusion_trainer",
                    "training_folder": str(output_root),
                    "sqlite_db_path": str((repo_root / "aitk_db.db").resolve()),
                    "device": "cuda",
                    "trigger_word": settings["trigger_word"],
                    "performance_log_every": settings["performance_log_every"],
                    "disable_progress_bar": settings["disable_progress_bar"],
                    "progress_bar_mininterval": settings["progress_bar_mininterval"],
                    "network": {
                        "type": "lora",
                        "linear": settings["rank"],
                        "linear_alpha": settings["rank"],
                        "conv": settings["conv_rank"],
                        "conv_alpha": settings["conv_rank"],
                        "lokr_full_rank": True,
                        "lokr_factor": -1,
                        "network_kwargs": {
                            "ignore_if_contains": [],
                        },
                    },
                    "save": {
                        "dtype": settings["save_dtype"],
                        "save_every": settings["save_every"],
                        "save_best_model": settings["save_best_model"],
                        "max_step_saves_to_keep": 4,
                        "save_format": settings["save_format"],
                        "push_to_hub": False,
                    },
                    "datasets": [
                        {
                            "folder_path": dataset_path,
                            "cache_dir": str(dataset_cache_dir),
                            "mask_path": None,
                            "mask_min_value": 0.1,
                            "default_caption": "",
                            "caption_ext": settings["caption_ext"],
                            "caption_dropout_rate": 0.05,
                            "cache_latents_to_disk": settings["cache_latents_to_disk"],
                            "is_reg": False,
                            "network_weight": 1,
                            "resolution": settings["resolution"],
                            "controls": [],
                            "shrink_video_to_frames": False,
                            "num_frames": 1,
                            "do_i2v": False,
                            "flip_x": False,
                            "flip_y": False,
                        }
                    ],
                    "train": {
                        "batch_size": settings["batch_size"],
                        "bypass_guidance_embedding": False,
                        "steps": settings["steps"],
                        "gradient_accumulation": settings["gradient_accumulation"],
                        "train_unet": True,
                        "train_text_encoder": settings["train_text_encoder"],
                        "gradient_checkpointing": True,
                        "noise_scheduler": "ddpm",
                        "optimizer": settings["optimizer"],
                        "timestep_type": "sigmoid",
                        "content_or_style": "balanced",
                        "optimizer_params": {
                            "weight_decay": 1e-4,
                        },
                        "unload_text_encoder": False,
                        "cache_text_embeddings": False,
                        "lr": settings["lr"],
                        "ema_config": {
                            "use_ema": False,
                            "ema_decay": 0.99,
                        },
                        "skip_first_sample": not settings["no_skip_first_sample"],
                        "force_first_sample": False,
                        "disable_sampling": disable_sampling,
                        "dtype": settings["dtype"],
                        "diff_output_preservation": False,
                        "diff_output_preservation_multiplier": 1.0,
                        "diff_output_preservation_class": "person",
                        "switch_boundary_every": 1,
                        "loss_type": "mse",
                    },
                    "logging": {
                        "log_every": 1,
                        "use_ui_logger": False,
                    },
                    "model": {
                        "name_or_path": model_path,
                        "quantize": False,
                        "qtype": "qfloat8",
                        "quantize_te": False,
                        "qtype_te": "qfloat8",
                        "arch": "sdxl",
                        "low_vram": settings["low_vram"],
                        "model_kwargs": {},
                    },
                    "sample": {
                        "sampler": "ddpm",
                        "sample_every": settings["sample_every"],
                        "width": settings["sample_width"],
                        "height": settings["sample_height"],
                        "samples": sample_items,
                        "neg": settings["sample_neg"],
                        "seed": settings["seed"],
                        "walk_seed": settings["walk_seed"],
                        "guidance_scale": settings["guidance_scale"],
                        "sample_steps": settings["sample_steps"],
                        "num_frames": 1,
                        "fps": 1,
                    },
                }
            ],
        },
        "meta": {
            "name": "[name]",
            "version": "1.0",
        },
    }


def write_config(config: dict[str, Any], config_path: Path) -> Path:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    return config_path


def run_training(repo_root: Path, config_path: Path, log_path: str | None) -> int:
    command = [sys.executable, str(repo_root / "run.py"), str(config_path)]
    if log_path:
        command.extend(["--log", log_path])
    print("Launching:", " ".join(json.dumps(part) for part in command))
    result = subprocess.run(command, cwd=str(repo_root))
    return result.returncode


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent
    settings = merge_settings(args)
    config = build_config(settings, repo_root)

    if settings["config_path"]:
        config_path = Path(settings["config_path"]).expanduser().resolve()
    else:
        config_path = Path(settings["output"]).expanduser().resolve() / settings["name"] / "cli_job_config.json"

    write_config(config, config_path)
    print(f"Config written to: {config_path}")

    if settings["write_config_only"]:
        return 0

    return run_training(repo_root, config_path, settings["log"])


if __name__ == "__main__":
    raise SystemExit(main())

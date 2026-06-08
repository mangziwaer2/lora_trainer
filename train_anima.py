#!/usr/bin/env python3
import argparse
import json
import os
import re
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
SUPPORTED_WEIGHT_EXTENSIONS = (".safetensors", ".ckpt", ".pt")
IS_KAGGLE = bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE")) or Path("/kaggle").exists()
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "anima_lora_cli.example.yaml"
DEFAULT_SD_SCRIPTS_CANDIDATES = [
    REPO_ROOT / "vendor" / "sd-scripts",
    REPO_ROOT / "temp" / "sd-scripts",
]


DEFAULTS: dict[str, Any] = {
    "caption_ext": "txt",
    "caption_dropout_rate": 0.0,
    "caption_dropout_every_n_epochs": 0,
    "caption_tag_dropout_rate": 0.0,
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
    "unet_lr": None,
    "text_encoder_lr": None,
    "optimizer_type": "AdamW8bit",
    "lr_scheduler": "constant",
    "timestep_sampling": "sigmoid",
    "discrete_flow_shift": 1.0,
    "sigmoid_scale": 1.0,
    "weighting_scheme": "uniform",
    "logit_mean": 0.0,
    "logit_std": 1.0,
    "mode_scale": 1.29,
    "max_train_epochs": None,
    "max_train_steps": None,
    "save_every_n_epochs": 1,
    "save_every_n_steps": None,
    "mixed_precision": "bf16",
    "gradient_checkpointing": True,
    "cache_latents": True,
    "cache_latents_to_disk": False,
    "cache_text_encoder_outputs": True,
    "cache_text_encoder_outputs_to_disk": False,
    "network_train_unet_only": True,
    "network_train_text_encoder_only": False,
    "low_vram": False,
    "vae_chunk_size": 64,
    "vae_batch_size": None,
    "vae_disable_cache": True,
    "text_encoder_batch_size": None,
    "skip_cache_check": False,
    "save_model_as": "safetensors",
    "llm_adapter_path": None,
    "llm_adapter_lr": None,
    "self_attn_lr": None,
    "cross_attn_lr": None,
    "mlp_lr": None,
    "mod_lr": None,
    "t5_tokenizer_path": None,
    "qwen3_max_token_length": 512,
    "t5_max_token_length": 512,
    "network_dropout": None,
    "network_args": [],
    "network_reg_dims": None,
    "network_reg_lrs": None,
    "train_llm_adapter": False,
    "exclude_patterns": None,
    "include_patterns": None,
    "rank_dropout": None,
    "module_dropout": None,
    "loraplus_lr_ratio": None,
    "loraplus_unet_lr_ratio": None,
    "loraplus_text_encoder_lr_ratio": None,
    "network_weights": None,
    "auto_resume": True,
    "dim_from_weights": False,
    "base_weights": [],
    "base_weights_multiplier": [],
    "resume": None,
    "save_state": False,
    "save_state_on_train_end": False,
    "initial_epoch": None,
    "initial_step": None,
    "skip_until_initial_step": False,
    "validation_seed": None,
    "validation_split": 0.0,
    "validate_every_n_steps": None,
    "validate_every_n_epochs": None,
    "max_validation_steps": None,
    "blocks_to_swap": None,
    "unsloth_offload_checkpointing": False,
    "cpu_offload_checkpointing": False,
    "extra_args": [],
    "command_path": None,
    "sample_every": None,
    "sample_every_n_epochs": None,
    "sample_prompts": [],
    "sample_prompts_file": None,
    "sample_neg": "",
    "sample_width": 1024,
    "sample_height": 1024,
    "sample_steps": 20,
    "guidance_scale": 6.0,
    "sample_seed": 42,
    "sample_flow_shift": 3.0,
    "walk_seed": False,
    "sample_at_first": False,
    "disable_sampling": False,
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
        "--caption-dropout-rate",
        type=float,
        default=None,
        help="Caption dropout rate. Set 0 to allow text encoder output caching.",
    )
    parser.add_argument(
        "--caption-dropout-every-n-epochs",
        type=int,
        default=None,
        help="Drop all captions every N epochs.",
    )
    parser.add_argument(
        "--caption-tag-dropout-rate",
        type=float,
        default=None,
        help="Dropout rate for comma-separated caption tags.",
    )
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
    parser.add_argument("--network-dropout", type=float, default=None, help="LoRA neuron dropout.")
    parser.add_argument(
        "--network-arg",
        dest="network_args",
        action="append",
        default=None,
        help="Repeat to add network key=value arguments passed to networks.lora_anima.",
    )
    parser.add_argument("--network-reg-dims", default=None, help="Regex rank overrides for LoRA modules.")
    parser.add_argument("--network-reg-lrs", default=None, help="Regex LR overrides for LoRA modules.")
    parser.add_argument(
        "--train-llm-adapter",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train the Anima LLM adapter inside the LoRA network.",
    )
    parser.add_argument(
        "--exclude-patterns",
        default=None,
        help="Python-list string of regex patterns to exclude from LoRA injection.",
    )
    parser.add_argument(
        "--include-patterns",
        default=None,
        help="Python-list string of regex patterns to include for LoRA injection.",
    )
    parser.add_argument("--rank-dropout", type=float, default=None, help="LoRA rank dropout.")
    parser.add_argument("--module-dropout", type=float, default=None, help="LoRA module dropout.")
    parser.add_argument("--loraplus-lr-ratio", type=float, default=None, help="LoRA+ global LR ratio.")
    parser.add_argument("--loraplus-unet-lr-ratio", type=float, default=None, help="LoRA+ DiT-side LR ratio.")
    parser.add_argument(
        "--loraplus-text-encoder-lr-ratio",
        type=float,
        default=None,
        help="LoRA+ text-encoder-side LR ratio.",
    )
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=None, help="Learning rate.")
    parser.add_argument("--unet-lr", type=float, default=None, help="Optional DiT/UNet-side learning rate override.")
    parser.add_argument(
        "--text-encoder-lr",
        type=float,
        nargs="*",
        default=None,
        help="Optional text encoder learning rate override.",
    )
    parser.add_argument("--optimizer-type", "--optimizer", dest="optimizer_type", default=None, help="Optimizer type.")
    parser.add_argument("--lr-scheduler", default=None, help="Learning-rate scheduler.")
    parser.add_argument("--timestep-sampling", default=None, help="Anima timestep sampling mode.")
    parser.add_argument("--discrete-flow-shift", type=float, default=None, help="Discrete flow shift.")
    parser.add_argument("--sigmoid-scale", type=float, default=None, help="Scale for sigmoid timestep sampling.")
    parser.add_argument("--weighting-scheme", default=None, help="Loss/timestep weighting scheme.")
    parser.add_argument("--logit-mean", type=float, default=None, help="Logit-normal weighting mean.")
    parser.add_argument("--logit-std", type=float, default=None, help="Logit-normal weighting std.")
    parser.add_argument("--mode-scale", type=float, default=None, help="Mode weighting scale.")
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
        "--cache-latents-to-disk",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache latents to disk through sd-scripts.",
    )
    parser.add_argument(
        "--cache-text-encoder-outputs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache frozen text encoder outputs in sd-scripts.",
    )
    parser.add_argument(
        "--cache-text-encoder-outputs-to-disk",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Cache text encoder outputs to disk through sd-scripts.",
    )
    parser.add_argument(
        "--network-train-unet-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train only the Anima DiT/UNet-side LoRA. This should stay enabled when caching text encoder outputs.",
    )
    parser.add_argument(
        "--network-train-text-encoder-only",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Train only the text-encoder side of the LoRA network.",
    )
    parser.add_argument(
        "--low-vram",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Forward sd-scripts --lowram to reduce peak memory usage.",
    )
    parser.add_argument("--vae-chunk-size", type=int, default=None, help="VAE chunk size.")
    parser.add_argument("--vae-batch-size", type=int, default=None, help="VAE batch size for latent caching/encoding.")
    parser.add_argument(
        "--vae-disable-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Disable the VAE internal cache.",
    )
    parser.add_argument("--text-encoder-batch-size", type=int, default=None, help="Batch size for text encoder output caching.")
    parser.add_argument(
        "--skip-cache-check",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip cache validation checks in sd-scripts.",
    )
    parser.add_argument("--save-model-as", "--save-format", dest="save_model_as", default=None, choices=["safetensors", "ckpt", "pt"], help="Output format.")
    parser.add_argument("--network-weights", default=None, help="Optional pretrained LoRA/network weights.")
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Automatically reuse the latest state or LoRA weight found in the output directory when resume/network_weights is not set.",
    )
    parser.add_argument(
        "--dim-from-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Infer LoRA rank from network_weights.",
    )
    parser.add_argument("--resume", default=None, help="Path to a saved training state directory for full resume.")
    parser.add_argument(
        "--save-state",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save optimizer/scheduler/training state alongside checkpoints.",
    )
    parser.add_argument(
        "--save-state-on-train-end",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save optimizer/scheduler/training state at the end of training.",
    )
    parser.add_argument("--initial-epoch", type=int, default=None, help="Optional initial epoch override.")
    parser.add_argument("--initial-step", type=int, default=None, help="Optional initial global-step override.")
    parser.add_argument(
        "--skip-until-initial-step",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Skip dataloader steps until initial_step is reached.",
    )
    parser.add_argument(
        "--base-weight",
        dest="base_weights",
        action="append",
        default=None,
        help="Repeat to merge base LoRA/network weights before training.",
    )
    parser.add_argument(
        "--base-weights-multiplier",
        type=float,
        nargs="*",
        default=None,
        help="Optional multipliers for base_weight items.",
    )
    parser.add_argument("--validation-seed", type=int, default=None, help="Validation dataset shuffle seed.")
    parser.add_argument("--validation-split", type=float, default=None, help="Validation split fraction.")
    parser.add_argument("--validate-every-n-steps", type=int, default=None, help="Run validation every N steps.")
    parser.add_argument("--validate-every-n-epochs", type=int, default=None, help="Run validation every N epochs.")
    parser.add_argument("--max-validation-steps", type=int, default=None, help="Maximum validation items per validation run.")
    parser.add_argument("--blocks-to-swap", type=int, default=None, help="Optional block swapping count for VRAM reduction.")
    parser.add_argument(
        "--unsloth-offload-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable unsloth CPU offload checkpointing when supported.",
    )
    parser.add_argument(
        "--cpu-offload-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable CPU offload checkpointing when supported.",
    )
    parser.add_argument("--llm-adapter-lr", type=float, default=None, help="Optional LLM adapter learning rate.")
    parser.add_argument("--self-attn-lr", type=float, default=None, help="Optional self-attention learning rate.")
    parser.add_argument("--cross-attn-lr", type=float, default=None, help="Optional cross-attention learning rate.")
    parser.add_argument("--mlp-lr", type=float, default=None, help="Optional MLP learning rate.")
    parser.add_argument("--mod-lr", type=float, default=None, help="Optional AdaLN modulation learning rate.")
    parser.add_argument("--command-path", default=None, help="Optional path to write the generated launch command JSON.")
    parser.add_argument("--sample-every", type=int, default=None, help="Generate sample images every N steps.")
    parser.add_argument("--sample-every-n-epochs", type=int, default=None, help="Generate sample images every N epochs.")
    parser.add_argument("--sample-prompt", dest="sample_prompts", action="append", default=None, help="Repeat to add sample prompts.")
    parser.add_argument("--sample-prompts-file", default=None, help="Optional text/json/toml prompt file passed to sd-scripts.")
    parser.add_argument("--sample-neg", default=None, help="Negative prompt used for generated samples.")
    parser.add_argument("--sample-width", type=int, default=None, help="Sample image width.")
    parser.add_argument("--sample-height", type=int, default=None, help="Sample image height.")
    parser.add_argument("--sample-steps", type=int, default=None, help="Sample inference steps.")
    parser.add_argument("--guidance-scale", type=float, default=None, help="CFG scale for sampling.")
    parser.add_argument("--sample-seed", type=int, default=None, help="Seed used for sample generation.")
    parser.add_argument("--walk-seed", action=argparse.BooleanOptionalAction, default=None, help="Increment seed for each generated sample.")
    parser.add_argument("--sample-flow-shift", type=float, default=None, help="Flow shift used for sample generation.")
    parser.add_argument("--sample-at-first", action="store_true", default=None, help="Generate samples before training starts.")
    parser.add_argument("--disable-sampling", action="store_true", default=None, help="Disable sample generation.")
    parser.add_argument("--qwen3-max-token-length", type=int, default=None, help="Maximum Qwen3 token length.")
    parser.add_argument("--t5-max-token-length", type=int, default=None, help="Maximum T5 token length.")
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


def normalize_float_list(raw_value: Any, key: str) -> list[float] | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, float)):
        return [float(raw_value)]
    if isinstance(raw_value, list):
        if not all(isinstance(item, (int, float)) for item in raw_value):
            raise ValueError(f"{key} must be a number or a list of numbers")
        return [float(item) for item in raw_value]
    raise ValueError(f"{key} must be a number or a list of numbers")


def normalize_string_list(raw_value: Any, key: str) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, str):
        return [raw_value]
    if isinstance(raw_value, list):
        if not all(isinstance(item, str) for item in raw_value):
            raise ValueError(f"{key} must be a string or a list of strings")
        return list(raw_value)
    raise ValueError(f"{key} must be a string or a list of strings")


def normalize_mapping_arg(raw_value: Any, key: str, value_type: type[int] | type[float]) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        return raw_value
    if isinstance(raw_value, dict):
        parts: list[str] = []
        for map_key, map_value in raw_value.items():
            if not isinstance(map_key, str):
                raise ValueError(f"{key} keys must be strings")
            if not isinstance(map_value, (int, float)):
                raise ValueError(f"{key} values must be numeric")
            cast_value = int(map_value) if value_type is int else float(map_value)
            parts.append(f"{map_key}={cast_value}")
        return ",".join(parts)
    raise ValueError(f"{key} must be a string or a mapping")


def normalize_pattern_arg(raw_value: Any, key: str) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        return raw_value
    if isinstance(raw_value, list):
        if not all(isinstance(item, str) for item in raw_value):
            raise ValueError(f"{key} must be a string or a list of strings")
        return json.dumps(raw_value, ensure_ascii=False)
    raise ValueError(f"{key} must be a string or a list of strings")


def append_arg(command: list[str], flag: str, value: Any) -> None:
    if value is None:
        return
    command.extend([flag, str(value)])


def append_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def build_network_args(settings: dict[str, Any]) -> list[str]:
    network_args = [str(item) for item in settings["network_args"]]

    if settings["train_llm_adapter"]:
        network_args.append("train_llm_adapter=true")
    if settings["network_reg_dims"] is not None:
        network_args.append(f"network_reg_dims={settings['network_reg_dims']}")
    if settings["network_reg_lrs"] is not None:
        network_args.append(f"network_reg_lrs={settings['network_reg_lrs']}")
    if settings["exclude_patterns"] is not None:
        network_args.append(f"exclude_patterns={settings['exclude_patterns']}")
    if settings["include_patterns"] is not None:
        network_args.append(f"include_patterns={settings['include_patterns']}")
    if settings["rank_dropout"] is not None:
        network_args.append(f"rank_dropout={settings['rank_dropout']}")
    if settings["module_dropout"] is not None:
        network_args.append(f"module_dropout={settings['module_dropout']}")
    if settings["loraplus_lr_ratio"] is not None:
        network_args.append(f"loraplus_lr_ratio={settings['loraplus_lr_ratio']}")
    if settings["loraplus_unet_lr_ratio"] is not None:
        network_args.append(f"loraplus_unet_lr_ratio={settings['loraplus_unet_lr_ratio']}")
    if settings["loraplus_text_encoder_lr_ratio"] is not None:
        network_args.append(f"loraplus_text_encoder_lr_ratio={settings['loraplus_text_encoder_lr_ratio']}")

    return network_args


def build_sample_prompt_item(prompt: str, settings: dict[str, Any], prompt_index: int) -> dict[str, Any]:
    sample_seed = settings["sample_seed"]
    if sample_seed is not None and settings["walk_seed"]:
        sample_seed = int(sample_seed) + prompt_index

    return {
        "prompt": prompt,
        "negative_prompt": settings["sample_neg"],
        "width": int(settings["sample_width"]),
        "height": int(settings["sample_height"]),
        "sample_steps": int(settings["sample_steps"]),
        "scale": float(settings["guidance_scale"]),
        "seed": int(sample_seed) if sample_seed is not None else None,
        "flow_shift": float(settings["sample_flow_shift"]),
    }


def get_resume_search_dirs(output_root: Path, job_name: str) -> list[Path]:
    candidates = [output_root, output_root / job_name]
    unique_dirs: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            resolved = str(path.resolve())
        except FileNotFoundError:
            resolved = str(path)
        if resolved in seen:
            continue
        seen.add(resolved)
        if path.exists() and path.is_dir():
            unique_dirs.append(path)
    return unique_dirs


def read_train_state_metadata(state_dir: Path) -> tuple[int, int]:
    train_state_file = state_dir / "train_state.json"
    if train_state_file.exists():
        try:
            payload = json.loads(train_state_file.read_text(encoding="utf-8"))
            return int(payload.get("current_step", 0)), int(payload.get("current_epoch", 0))
        except Exception:
            pass

    step_match = re.match(r".*-step(\d{8})-state$", state_dir.name)
    if step_match:
        return int(step_match.group(1)), 0

    epoch_match = re.match(r".*-(\d{6})-state$", state_dir.name)
    if epoch_match:
        return 0, int(epoch_match.group(1))

    return 0, 0


def find_latest_resume_state(output_root: Path, job_name: str) -> Path | None:
    state_candidates: list[tuple[int, int, int, Path]] = []
    step_pattern = re.compile(rf"^{re.escape(job_name)}-step\d{{8}}-state$")
    epoch_pattern = re.compile(rf"^{re.escape(job_name)}-\d{{6}}-state$")
    last_name = f"{job_name}-state"

    for search_dir in get_resume_search_dirs(output_root, job_name):
        for path in search_dir.iterdir():
            if not path.is_dir():
                continue
            if path.name != last_name and not step_pattern.fullmatch(path.name) and not epoch_pattern.fullmatch(path.name):
                continue
            step_no, epoch_no = read_train_state_metadata(path)
            state_candidates.append((step_no, epoch_no, path.stat().st_mtime_ns, path))

    if not state_candidates:
        return None

    state_candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return state_candidates[0][3]


def find_latest_network_weight(output_root: Path, job_name: str) -> Path | None:
    weight_candidates: list[tuple[int, int, int, int, Path]] = []
    step_pattern = re.compile(rf"^{re.escape(job_name)}-step(\d{{8}})(\.[^.]+)$")
    epoch_pattern = re.compile(rf"^{re.escape(job_name)}-(\d{{6}})(\.[^.]+)$")

    for search_dir in get_resume_search_dirs(output_root, job_name):
        for path in search_dir.iterdir():
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_WEIGHT_EXTENSIONS:
                continue

            step_match = step_pattern.fullmatch(path.name)
            if step_match:
                weight_candidates.append((2, int(step_match.group(1)), 0, path.stat().st_mtime_ns, path))
                continue

            epoch_match = epoch_pattern.fullmatch(path.name)
            if epoch_match:
                weight_candidates.append((1, 0, int(epoch_match.group(1)), path.stat().st_mtime_ns, path))
                continue

            if path.stem == job_name:
                weight_candidates.append((3, 0, 0, path.stat().st_mtime_ns, path))

    if not weight_candidates:
        return None

    weight_candidates.sort(key=lambda item: (item[3], item[0], item[1], item[2]), reverse=True)
    return weight_candidates[0][4]


def resolve_auto_resume(settings: dict[str, Any], output_root: Path) -> dict[str, Any]:
    result = {
        "enabled": bool(settings.get("auto_resume", True)),
        "mode": "disabled",
        "resume_path": None,
        "network_weights_path": None,
        "dim_from_weights_auto_set": False,
    }

    if not result["enabled"]:
        return result

    if settings.get("resume"):
        result["mode"] = "manual_resume"
        result["resume_path"] = settings["resume"]
        return result

    if settings.get("network_weights"):
        result["mode"] = "manual_network_weights"
        result["network_weights_path"] = settings["network_weights"]
        return result

    latest_state_dir = find_latest_resume_state(output_root, settings["name"])
    if latest_state_dir is not None:
        settings["resume"] = str(latest_state_dir)
        result["mode"] = "auto_resume_state"
        result["resume_path"] = str(latest_state_dir)
        return result

    latest_weight = find_latest_network_weight(output_root, settings["name"])
    if latest_weight is not None:
        settings["network_weights"] = str(latest_weight)
        if not settings.get("dim_from_weights", False):
            settings["dim_from_weights"] = True
            result["dim_from_weights_auto_set"] = True
        result["mode"] = "auto_resume_weights"
        result["network_weights_path"] = str(latest_weight)
        return result

    result["mode"] = "no_checkpoint_found"
    return result


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

    settings["text_encoder_lr"] = normalize_float_list(settings.get("text_encoder_lr"), "text_encoder_lr")
    settings["base_weights"] = normalize_string_list(settings.get("base_weights"), "base_weights")
    settings["base_weights_multiplier"] = normalize_float_list(settings.get("base_weights_multiplier"), "base_weights_multiplier") or []
    settings["network_args"] = normalize_string_list(settings.get("network_args"), "network_args")
    settings["extra_args"] = normalize_string_list(settings.get("extra_args"), "extra_args")
    settings["sample_prompts"] = normalize_string_list(settings.get("sample_prompts"), "sample_prompts")
    settings["network_reg_dims"] = normalize_mapping_arg(settings.get("network_reg_dims"), "network_reg_dims", int)
    settings["network_reg_lrs"] = normalize_mapping_arg(settings.get("network_reg_lrs"), "network_reg_lrs", float)
    settings["exclude_patterns"] = normalize_pattern_arg(settings.get("exclude_patterns"), "exclude_patterns")
    settings["include_patterns"] = normalize_pattern_arg(settings.get("include_patterns"), "include_patterns")
    if settings["cache_latents_to_disk"] and not settings["cache_latents"]:
        settings["cache_latents"] = True
    if settings["cache_text_encoder_outputs_to_disk"] and not settings["cache_text_encoder_outputs"]:
        settings["cache_text_encoder_outputs"] = True

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
    if settings["sample_every"] is not None and settings["sample_every"] < 1:
        raise ValueError("sample_every must be >= 1 when provided")
    if settings["sample_every_n_epochs"] is not None and settings["sample_every_n_epochs"] < 1:
        raise ValueError("sample_every_n_epochs must be >= 1 when provided")
    if settings["sample_width"] < 64 or settings["sample_height"] < 64:
        raise ValueError("sample_width and sample_height must be >= 64")
    if settings["sample_steps"] < 1:
        raise ValueError("sample_steps must be >= 1")
    if settings["caption_dropout_rate"] < 0 or settings["caption_dropout_rate"] > 1:
        raise ValueError("caption_dropout_rate must be between 0 and 1")
    if settings["caption_tag_dropout_rate"] < 0 or settings["caption_tag_dropout_rate"] > 1:
        raise ValueError("caption_tag_dropout_rate must be between 0 and 1")
    if settings["caption_dropout_every_n_epochs"] < 0:
        raise ValueError("caption_dropout_every_n_epochs must be >= 0")
    if settings["validation_split"] < 0 or settings["validation_split"] > 1:
        raise ValueError("validation_split must be between 0 and 1")
    if settings["vae_batch_size"] is not None and settings["vae_batch_size"] < 1:
        raise ValueError("vae_batch_size must be >= 1 when provided")
    if settings["text_encoder_batch_size"] is not None and settings["text_encoder_batch_size"] < 1:
        raise ValueError("text_encoder_batch_size must be >= 1 when provided")
    if settings["blocks_to_swap"] is not None and settings["blocks_to_swap"] < 1:
        raise ValueError("blocks_to_swap must be >= 1 when provided")
    if settings["validate_every_n_steps"] is not None and settings["validate_every_n_steps"] < 1:
        raise ValueError("validate_every_n_steps must be >= 1 when provided")
    if settings["validate_every_n_epochs"] is not None and settings["validate_every_n_epochs"] < 1:
        raise ValueError("validate_every_n_epochs must be >= 1 when provided")
    if settings["max_validation_steps"] is not None and settings["max_validation_steps"] < 1:
        raise ValueError("max_validation_steps must be >= 1 when provided")
    if settings["initial_epoch"] is not None and settings["initial_epoch"] < 1:
        raise ValueError("initial_epoch must be >= 1 when provided")
    if settings["initial_step"] is not None and settings["initial_step"] < 0:
        raise ValueError("initial_step must be >= 0 when provided")
    if settings["network_dropout"] is not None and not 0 <= settings["network_dropout"] <= 1:
        raise ValueError("network_dropout must be between 0 and 1 when provided")
    if settings["rank_dropout"] is not None and not 0 <= settings["rank_dropout"] < 1:
        raise ValueError("rank_dropout must be between 0 and 1 when provided")
    if settings["module_dropout"] is not None and not 0 <= settings["module_dropout"] < 1:
        raise ValueError("module_dropout must be between 0 and 1 when provided")
    if len(settings["base_weights_multiplier"]) not in {0, len(settings["base_weights"])}:
        raise ValueError("base_weights_multiplier must be empty or have the same length as base_weights")
    if settings["network_train_unet_only"] and settings["network_train_text_encoder_only"]:
        raise ValueError("network_train_unet_only and network_train_text_encoder_only cannot both be enabled")
    if settings["cache_text_encoder_outputs"] and settings["network_train_text_encoder_only"]:
        raise ValueError("cache_text_encoder_outputs cannot be enabled while training only the text encoder")
    if settings["cache_text_encoder_outputs"] and not settings["network_train_unet_only"]:
        raise ValueError("cache_text_encoder_outputs requires network_train_unet_only=true for Anima")
    if settings["cache_text_encoder_outputs"] and settings["caption_tag_dropout_rate"] > 0:
        raise ValueError("caption_tag_dropout_rate cannot be used with cache_text_encoder_outputs")
    if settings["cache_text_encoder_outputs"] and settings["shuffle_caption"]:
        raise ValueError("shuffle_caption cannot be used with cache_text_encoder_outputs")

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


def load_sample_prompts(settings: dict[str, Any]) -> list[str]:
    prompts = list(settings.get("sample_prompts", []))
    return prompts


def build_sample_prompts_path(settings: dict[str, Any], job_root: Path) -> Path | None:
    if settings["disable_sampling"]:
        return None

    sample_prompts_file = settings.get("sample_prompts_file")
    if sample_prompts_file:
        prompt_file_path = resolve_path(sample_prompts_file, base_dir=get_base_dir_for_setting(settings, "sample_prompts_file"))
        if not prompt_file_path.exists():
            raise FileNotFoundError(f"Sample prompts file does not exist: {sample_prompts_file}")
        if prompt_file_path.suffix.lower() in {".json", ".toml"}:
            return prompt_file_path

        file_prompts = [line.strip() for line in prompt_file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(file_prompts) == 0:
            return None

        payload = [build_sample_prompt_item(prompt, settings, prompt_index) for prompt_index, prompt in enumerate(file_prompts)]
        prompts_path = job_root / "anima_sample_prompts.json"
        prompts_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return prompts_path

    prompts = load_sample_prompts(settings)
    if len(prompts) == 0:
        return None

    payload = [build_sample_prompt_item(prompt, settings, prompt_index) for prompt_index, prompt in enumerate(prompts)]
    prompts_path = job_root / "anima_sample_prompts.json"
    prompts_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return prompts_path


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
                "validation_seed": settings["validation_seed"],
                "validation_split": float(settings["validation_split"]),
                "subsets": [
                    {
                        "image_dir": str(path),
                        "num_repeats": int(settings["num_repeats"]),
                        "flip_aug": bool(settings["flip_aug"]),
                        "caption_dropout_rate": float(settings["caption_dropout_rate"]),
                        "caption_dropout_every_n_epochs": int(settings["caption_dropout_every_n_epochs"]),
                        "caption_tag_dropout_rate": float(settings["caption_tag_dropout_rate"]),
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


def build_command(settings: dict[str, Any], dataset_config_path: Path, sample_prompts_path: Path | None) -> tuple[list[str], Path]:
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
            "--sigmoid_scale",
            str(settings["sigmoid_scale"]),
            "--weighting_scheme",
            settings["weighting_scheme"],
            "--logit_mean",
            str(settings["logit_mean"]),
            "--logit_std",
            str(settings["logit_std"]),
            "--mode_scale",
            str(settings["mode_scale"]),
            "--mixed_precision",
            settings["mixed_precision"],
            "--seed",
            str(int(settings["seed"])),
            "--vae_chunk_size",
            str(int(settings["vae_chunk_size"])),
            "--qwen3_max_token_length",
            str(int(settings["qwen3_max_token_length"])),
            "--t5_max_token_length",
            str(int(settings["t5_max_token_length"])),
        ]
    )

    append_arg(command, "--network_alpha", int(settings["network_alpha"]) if settings["network_alpha"] is not None else None)
    append_arg(command, "--network_dropout", settings["network_dropout"])
    append_arg(command, "--unet_lr", settings["unet_lr"])
    append_arg(command, "--validation_seed", settings["validation_seed"])
    append_arg(command, "--validation_split", settings["validation_split"] if settings["validation_split"] > 0 else None)
    append_arg(command, "--validate_every_n_steps", settings["validate_every_n_steps"])
    append_arg(command, "--validate_every_n_epochs", settings["validate_every_n_epochs"])
    append_arg(command, "--max_validation_steps", settings["max_validation_steps"])
    append_arg(command, "--blocks_to_swap", settings["blocks_to_swap"])
    append_arg(command, "--vae_batch_size", settings["vae_batch_size"])
    append_arg(command, "--text_encoder_batch_size", settings["text_encoder_batch_size"])
    append_arg(command, "--llm_adapter_lr", settings["llm_adapter_lr"])
    append_arg(command, "--self_attn_lr", settings["self_attn_lr"])
    append_arg(command, "--cross_attn_lr", settings["cross_attn_lr"])
    append_arg(command, "--mlp_lr", settings["mlp_lr"])
    append_arg(command, "--mod_lr", settings["mod_lr"])
    if settings["max_train_steps"] is not None:
        command.extend(["--max_train_steps", str(int(settings["max_train_steps"]))])
    elif settings["max_train_epochs"] is not None:
        command.extend(["--max_train_epochs", str(int(settings["max_train_epochs"]))])
    if settings["save_every_n_steps"] is not None:
        command.extend(["--save_every_n_steps", str(int(settings["save_every_n_steps"]))])
    elif settings["max_train_epochs"] is not None and settings["save_every_n_epochs"] is not None:
        command.extend(["--save_every_n_epochs", str(int(settings["save_every_n_epochs"]))])

    if settings["text_encoder_lr"] is not None:
        command.append("--text_encoder_lr")
        command.extend(str(item) for item in settings["text_encoder_lr"])

    network_args = build_network_args(settings)
    if len(network_args) > 0:
        command.append("--network_args")
        command.extend(network_args)

    if sample_prompts_path is not None:
        command.extend(["--sample_prompts", str(sample_prompts_path)])
        if settings["sample_every"] is not None:
            command.extend(["--sample_every_n_steps", str(int(settings["sample_every"]))])
        if settings["sample_every_n_epochs"] is not None:
            command.extend(["--sample_every_n_epochs", str(int(settings["sample_every_n_epochs"]))])
        if settings["sample_at_first"]:
            command.append("--sample_at_first")
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
    if settings["network_weights"] is not None:
        command.extend(
            [
                "--network_weights",
                normalize_existing_path(
                    settings["network_weights"],
                    "Network weights",
                    base_dir=get_base_dir_for_setting(settings, "network_weights"),
                ),
            ]
        )
    if settings["resume"] is not None:
        command.extend(
            [
                "--resume",
                normalize_existing_path(
                    settings["resume"],
                    "Resume state",
                    base_dir=get_base_dir_for_setting(settings, "resume"),
                ),
            ]
        )
    if len(settings["base_weights"]) > 0:
        command.append("--base_weights")
        command.extend(
            normalize_existing_path(
                item,
                "Base weight",
                base_dir=get_base_dir_for_setting(settings, "base_weights"),
            )
            for item in settings["base_weights"]
        )
    if len(settings["base_weights_multiplier"]) > 0:
        command.append("--base_weights_multiplier")
        command.extend(str(item) for item in settings["base_weights_multiplier"])

    if settings["gradient_checkpointing"]:
        command.append("--gradient_checkpointing")
    append_flag(command, "--cache_latents", settings["cache_latents"])
    append_flag(command, "--cache_latents_to_disk", settings["cache_latents_to_disk"])
    append_flag(command, "--cache_text_encoder_outputs", settings["cache_text_encoder_outputs"])
    append_flag(command, "--cache_text_encoder_outputs_to_disk", settings["cache_text_encoder_outputs_to_disk"])
    append_flag(command, "--network_train_unet_only", settings["network_train_unet_only"])
    append_flag(command, "--network_train_text_encoder_only", settings["network_train_text_encoder_only"])
    append_flag(command, "--lowram", settings["low_vram"])
    append_flag(command, "--vae_disable_cache", settings["vae_disable_cache"])
    append_flag(command, "--skip_cache_check", settings["skip_cache_check"])
    append_flag(command, "--dim_from_weights", settings["dim_from_weights"])
    append_flag(command, "--save_state", settings["save_state"])
    append_flag(command, "--save_state_on_train_end", settings["save_state_on_train_end"])
    append_flag(command, "--skip_until_initial_step", settings["skip_until_initial_step"])
    append_flag(command, "--unsloth_offload_checkpointing", settings["unsloth_offload_checkpointing"])
    append_flag(command, "--cpu_offload_checkpointing", settings["cpu_offload_checkpointing"])
    append_arg(command, "--initial_epoch", settings["initial_epoch"])
    append_arg(command, "--initial_step", settings["initial_step"])
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
    auto_resume_result = resolve_auto_resume(settings, output_root)
    if auto_resume_result["mode"] == "auto_resume_state":
        print(f"Auto-resume: using latest state {auto_resume_result['resume_path']}")
    elif auto_resume_result["mode"] == "auto_resume_weights":
        extra = " (dim_from_weights enabled)" if auto_resume_result["dim_from_weights_auto_set"] else ""
        print(f"Auto-resume: using latest LoRA weights {auto_resume_result['network_weights_path']}{extra}")

    image_dirs, dataset_root = collect_plain_dataset(settings)

    dataset_config_path = build_dataset_config(settings, image_dirs, job_root / "anima_dataset_config.toml")
    sample_prompts_path = build_sample_prompts_path(settings, job_root)
    command, workdir = build_command(settings, dataset_config_path, sample_prompts_path)

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
        "auto_resume": auto_resume_result,
        "resume_used": settings.get("resume"),
        "network_weights_used": settings.get("network_weights"),
        "sample_prompts_path": str(sample_prompts_path) if sample_prompts_path is not None else None,
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

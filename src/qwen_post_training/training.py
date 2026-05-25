"""Local SFT training helpers around MLX-LM LoRA."""

from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from qwen_post_training.data_pipeline import validate_split_dir
from qwen_post_training.inference import DEFAULT_MODEL, resolve_mlx_command


DEFAULT_SFT_PROFILE = "local_16gb"
SMOKE_SFT_PROFILE = "smoke"

DEFAULT_SFT_CONFIG = {
    "model": DEFAULT_MODEL,
    "fine_tune_type": "lora",
    "batch_size": 1,
    "grad_accumulation_steps": 8,
    "max_seq_length": 512,
    "num_layers": 4,
    "mask_prompt": True,
    "iters": 200,
    "val_batches": 10,
    "learning_rate": 1e-5,
    "steps_per_report": 1,
    "steps_per_eval": 50,
    "save_every": 50,
    "test": True,
    "test_batches": 10,
    "grad_checkpoint": True,
    "seed": 7,
    "lora_parameters": {
        "rank": 4,
        "dropout": 0.0,
        "scale": 20.0,
    },
}


@dataclass(frozen=True)
class SftTrainingRequest:
    dataset_dir: Path
    model: str = DEFAULT_MODEL
    run_id: Optional[str] = None
    adapters_root: Path = Path("adapters/sft")
    runs_root: Path = Path("runs/sft")
    config_path: Optional[Path] = Path("configs/sft.yaml")
    profile: Optional[str] = None
    iters: Optional[int] = None
    seed: int = 7
    smoke: bool = False


def utc_run_id(prefix: str = "sft") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}-{stamp}"


def shell_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def load_yaml_config(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only without train deps.
        raise RuntimeError(
            "Reading SFT YAML config requires PyYAML. Install with: "
            'pip install "mlx-lm[train]"'
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return loaded


def selected_profile_name(request: SftTrainingRequest) -> str:
    if request.profile:
        return request.profile
    if request.smoke:
        return SMOKE_SFT_PROFILE
    return DEFAULT_SFT_PROFILE


def training_section_for_profile(raw: Mapping[str, Any], profile_name: str) -> dict[str, Any]:
    model = raw.get("model", {})
    training = raw.get("training", {})
    if not isinstance(model, Mapping):
        model = {}
    if not isinstance(training, Mapping):
        training = {}

    merged_training = dict(training)
    profiles = raw.get("profiles", {})
    if profiles is not None and not isinstance(profiles, Mapping):
        raise ValueError("profiles must be a YAML object")
    if isinstance(profiles, Mapping) and profiles:
        profile = profiles.get(profile_name)
        if not isinstance(profile, Mapping):
            raise ValueError(
                f"SFT profile {profile_name!r} not found in config; "
                f"available profiles: {', '.join(sorted(str(name) for name in profiles))}"
            )
        merged_training.update(profile)
    return {
        "model": model,
        "training": merged_training,
    }


def flatten_sft_config(raw: Mapping[str, Any], profile_name: str = DEFAULT_SFT_PROFILE) -> dict[str, Any]:
    selected = training_section_for_profile(raw, profile_name)
    model = selected["model"]
    training = selected["training"]

    lora_parameters = dict(DEFAULT_SFT_CONFIG["lora_parameters"])
    if "lora_parameters" in training and isinstance(training["lora_parameters"], Mapping):
        lora_parameters.update(training["lora_parameters"])
    if "lora_rank" in training:
        lora_parameters["rank"] = training["lora_rank"]
    if "lora_dropout" in training:
        lora_parameters["dropout"] = training["lora_dropout"]
    if "lora_scale" in training:
        lora_parameters["scale"] = training["lora_scale"]

    config = dict(DEFAULT_SFT_CONFIG)
    config.update(
        {
            "model": model.get("base", DEFAULT_MODEL),
            "fine_tune_type": training.get("adapter", DEFAULT_SFT_CONFIG["fine_tune_type"]),
            "batch_size": training.get("batch_size", DEFAULT_SFT_CONFIG["batch_size"]),
            "grad_accumulation_steps": training.get(
                "gradient_accumulation_steps",
                DEFAULT_SFT_CONFIG["grad_accumulation_steps"],
            ),
            "max_seq_length": training.get(
                "max_seq_length",
                DEFAULT_SFT_CONFIG["max_seq_length"],
            ),
            "num_layers": training.get(
                "trainable_layers",
                DEFAULT_SFT_CONFIG["num_layers"],
            ),
            "mask_prompt": training.get(
                "mask_prompt_tokens",
                DEFAULT_SFT_CONFIG["mask_prompt"],
            ),
            "iters": training.get("iters", DEFAULT_SFT_CONFIG["iters"]),
            "val_batches": training.get(
                "val_batches",
                DEFAULT_SFT_CONFIG["val_batches"],
            ),
            "learning_rate": training.get(
                "learning_rate",
                DEFAULT_SFT_CONFIG["learning_rate"],
            ),
            "steps_per_report": training.get(
                "steps_per_report",
                DEFAULT_SFT_CONFIG["steps_per_report"],
            ),
            "steps_per_eval": training.get(
                "steps_per_eval",
                DEFAULT_SFT_CONFIG["steps_per_eval"],
            ),
            "save_every": training.get("save_every", DEFAULT_SFT_CONFIG["save_every"]),
            "test": training.get("test", DEFAULT_SFT_CONFIG["test"]),
            "test_batches": training.get(
                "test_batches",
                DEFAULT_SFT_CONFIG["test_batches"],
            ),
            "grad_checkpoint": training.get(
                "grad_checkpoint",
                DEFAULT_SFT_CONFIG["grad_checkpoint"],
            ),
            "seed": training.get("seed", DEFAULT_SFT_CONFIG["seed"]),
            "lora_parameters": lora_parameters,
        }
    )
    return config


def sft_config_for_request(request: SftTrainingRequest) -> dict[str, Any]:
    profile_name = selected_profile_name(request)
    config = flatten_sft_config(load_yaml_config(request.config_path), profile_name)
    config["model"] = request.model or config["model"]
    config["seed"] = request.seed
    if request.iters is not None:
        config["iters"] = request.iters
    if request.smoke:
        config["iters"] = min(int(config["iters"]), 10)
    return config


def write_mlx_lora_config(path: Path, config: Mapping[str, Any], data_dir: Path, adapter_dir: Path) -> None:
    payload = dict(config)
    payload.update(
        {
            "train": True,
            "data": str(data_dir),
            "adapter_path": str(adapter_dir),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def prepare_mlx_sft_data(source_dir: Path, target_dir: Path) -> None:
    errors = validate_split_dir(source_dir)
    if errors:
        raise ValueError("invalid SFT split directory: " + "; ".join(errors))

    target_dir.mkdir(parents=True, exist_ok=True)
    split_map = {
        "train.jsonl": "train.jsonl",
        "validation.jsonl": "valid.jsonl",
        "test.jsonl": "test.jsonl",
    }
    for source_name, target_name in split_map.items():
        shutil.copyfile(source_dir / source_name, target_dir / target_name)


def build_lora_command(config_path: Path) -> list[str]:
    base = resolve_mlx_command("mlx_lm.lora") or ["mlx_lm.lora"]
    return [*base, "--config", str(config_path)]


def write_run_metadata(
    path: Path,
    *,
    request: SftTrainingRequest,
    config: Mapping[str, Any],
    adapter_dir: Path,
    data_dir: Path,
    lora_config_path: Path,
    log_path: Path,
    metrics_path: Path,
    command: Sequence[str],
    dry_run: bool,
) -> None:
    metadata = {
        "run_id": request.run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "backend": "mlx_lm.lora",
        "profile": selected_profile_name(request),
        "model": config["model"],
        "dataset_dir": str(request.dataset_dir),
        "prepared_data_dir": str(data_dir),
        "adapter_dir": str(adapter_dir),
        "mlx_lora_config": str(lora_config_path),
        "training_log": str(log_path),
        "metrics": str(metrics_path),
        "command": list(command),
        "training": config,
    }
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_mlx_metrics(log_text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    val_matches = re.findall(r"Iter (\d+): Val loss ([0-9.]+)", log_text)
    train_matches = re.findall(r"Iter (\d+): Train loss ([0-9.]+).*?Peak mem ([0-9.]+) GB", log_text)
    test_match = re.search(r"Test loss ([0-9]+(?:\.[0-9]+)?), Test ppl ([0-9]+(?:\.[0-9]+)?)", log_text)

    if val_matches:
        iteration, loss = val_matches[-1]
        metrics["final_validation"] = {
            "iteration": int(iteration),
            "loss": float(loss),
        }
    if train_matches:
        iteration, loss, peak_mem_gb = train_matches[-1]
        metrics["final_train"] = {
            "iteration": int(iteration),
            "loss": float(loss),
            "peak_memory_gb": float(peak_mem_gb),
        }
    if test_match:
        metrics["test"] = {
            "loss": float(test_match.group(1)),
            "perplexity": float(test_match.group(2)),
        }
    return metrics


def run_command_with_log(command: Sequence[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return process.wait()


def run_sft_training(request: SftTrainingRequest, dry_run: bool = False) -> dict[str, Any]:
    run_id = request.run_id or utc_run_id()
    request = SftTrainingRequest(
        dataset_dir=request.dataset_dir,
        model=request.model,
        run_id=run_id,
        adapters_root=request.adapters_root,
        runs_root=request.runs_root,
        config_path=request.config_path,
        profile=selected_profile_name(request),
        iters=request.iters,
        seed=request.seed,
        smoke=request.smoke,
    )
    run_dir = request.runs_root / run_id
    adapter_dir = request.adapters_root / run_id
    data_dir = run_dir / "data"
    lora_config_path = run_dir / "mlx_lora_config.json"
    metadata_path = run_dir / "metadata.json"
    log_path = run_dir / "training.log"
    metrics_path = run_dir / "metrics.json"

    existing_paths = [path for path in (run_dir, adapter_dir) if path.exists()]
    if existing_paths:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"SFT run_id {run_id!r} already exists at {existing}; "
            "choose a unique --run-id before starting training"
        )

    run_dir.mkdir(parents=True, exist_ok=False)
    adapter_dir.mkdir(parents=True, exist_ok=False)
    prepare_mlx_sft_data(request.dataset_dir, data_dir)
    config = sft_config_for_request(request)
    write_mlx_lora_config(lora_config_path, config, data_dir, adapter_dir)
    command = build_lora_command(lora_config_path)
    write_run_metadata(
        metadata_path,
        request=request,
        config=config,
        adapter_dir=adapter_dir,
        data_dir=data_dir,
        lora_config_path=lora_config_path,
        log_path=log_path,
        metrics_path=metrics_path,
        command=command,
        dry_run=dry_run,
    )

    if dry_run:
        return {
            "run_id": run_id,
            "adapter_dir": str(adapter_dir),
            "run_dir": str(run_dir),
            "metadata": str(metadata_path),
            "training_log": str(log_path),
            "metrics": str(metrics_path),
            "command": shell_command(command),
            "dry_run": True,
        }

    if resolve_mlx_command("mlx_lm.lora") is None:
        raise RuntimeError(
            "MLX-LM LoRA training is not installed. Install with: "
            'pip install "mlx-lm[train]"'
        )

    returncode = run_command_with_log(command, log_path)
    if returncode != 0:
        raise RuntimeError(f"MLX-LM LoRA training exited with {returncode}")

    metrics = parse_mlx_metrics(log_path.read_text(encoding="utf-8"))
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return {
        "run_id": run_id,
        "adapter_dir": str(adapter_dir),
        "run_dir": str(run_dir),
        "metadata": str(metadata_path),
        "training_log": str(log_path),
        "metrics": str(metrics_path),
        "command": shell_command(command),
        "dry_run": False,
    }

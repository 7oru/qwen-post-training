"""DPO readiness checks and explicit local fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from qwen_post_training.inference import DEFAULT_MODEL


DPO_UNAVAILABLE_REASON = (
    "Current MLX-LM install does not expose a local DPO training command. "
    "Keep DPO post-Beta until a supported Apple Silicon backend is selected."
)


@dataclass(frozen=True)
class DpoTrainingRequest:
    config_path: Optional[Path] = Path("configs/dpo.yaml")
    run_id: Optional[str] = None
    runs_root: Path = Path("runs/dpo")
    model: str = DEFAULT_MODEL
    sft_adapter: Optional[str] = None


def utc_dpo_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"dpo-{stamp}"


def load_dpo_config(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}

    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - only without train extras.
        raise RuntimeError(
            "Reading DPO YAML config requires PyYAML. Install with: "
            'pip install "mlx-lm[train]"'
        ) from exc

    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a YAML object")
    return loaded


def dpo_readiness_report(request: DpoTrainingRequest) -> dict[str, Any]:
    raw = load_dpo_config(request.config_path)
    model_config = raw.get("model", {})
    data_config = raw.get("data", {})
    training_config = raw.get("training", {})
    if not isinstance(model_config, dict):
        model_config = {}
    if not isinstance(data_config, dict):
        data_config = {}
    if not isinstance(training_config, dict):
        training_config = {}

    return {
        "status": "unavailable",
        "reason": DPO_UNAVAILABLE_REASON,
        "backend": "mlx_lm",
        "model": request.model or model_config.get("base", DEFAULT_MODEL),
        "sft_adapter": request.sft_adapter or model_config.get("sft_adapter"),
        "data": {
            "train": data_config.get("train", "data/processed/dpo/train.jsonl"),
            "validation": data_config.get(
                "validation",
                "data/processed/dpo/validation.jsonl",
            ),
            "test": data_config.get("test", "data/processed/dpo/test.jsonl"),
        },
        "training": {
            "batch_size": training_config.get("batch_size", 1),
            "gradient_accumulation_steps": training_config.get(
                "gradient_accumulation_steps",
                8,
            ),
            "max_seq_length": training_config.get("max_seq_length", 512),
            "beta": training_config.get("beta", 0.1),
            "lora_rank": training_config.get("lora_rank", 4),
            "trainable_layers": training_config.get("trainable_layers", 4),
        },
    }


def run_dpo_check(request: DpoTrainingRequest, dry_run: bool = True) -> dict[str, Any]:
    run_id = request.run_id or utc_dpo_run_id()
    run_dir = request.runs_root / run_id
    metadata_path = run_dir / "metadata.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    report = dpo_readiness_report(request)
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "config": str(request.config_path) if request.config_path else None,
        **report,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metadata": str(metadata_path),
        **report,
    }

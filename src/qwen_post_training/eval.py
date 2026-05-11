"""Recorded local eval runs for baseline and adapter comparisons."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from qwen_post_training.data_pipeline import read_jsonl, write_jsonl
from qwen_post_training.inference import DEFAULT_MODEL, GenerationRequest, generate


@dataclass(frozen=True)
class EvalRequest:
    prompts_path: Path
    run_id: Optional[str] = None
    output_root: Path = Path("runs/eval")
    model: str = DEFAULT_MODEL
    adapter_path: Optional[str] = None
    max_tokens: int = 256
    temperature: float = 0.0
    seed: Optional[int] = 7
    backend: str = "mlx"


def utc_eval_run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"eval-{stamp}"


def load_eval_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for index, record in enumerate(read_jsonl(path), start=1):
        prompt = record.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}:{index}: prompt must be a non-empty string")
        prompts.append(prompt.strip())
    if not prompts:
        raise ValueError(f"{path} does not contain any prompts")
    return prompts


def run_eval(request: EvalRequest, dry_run: bool = False) -> dict[str, Any]:
    run_id = request.run_id or utc_eval_run_id()
    run_dir = request.output_root / run_id
    results_path = run_dir / "results.jsonl"
    metadata_path = run_dir / "metadata.json"
    prompts = load_eval_prompts(request.prompts_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts, start=1):
        generation_request = GenerationRequest(
            prompt=prompt,
            model=request.model,
            adapter_path=request.adapter_path,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            seed=request.seed,
            backend=request.backend,
        )
        results.append(
            {
                "index": index,
                "prompt": prompt,
                "response": generate(generation_request, dry_run=dry_run),
            }
        )

    write_jsonl(results_path, results)
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "prompts": str(request.prompts_path),
        "results": str(results_path),
        "model": request.model,
        "adapter_path": request.adapter_path,
        "backend": request.backend,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "seed": request.seed,
        "prompt_count": len(prompts),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "metadata": str(metadata_path),
        "results": str(results_path),
        "prompt_count": len(prompts),
        "dry_run": dry_run,
    }

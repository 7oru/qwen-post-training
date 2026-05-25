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


def load_eval_results(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path} does not contain any eval results")
    for index, row in enumerate(rows, start=1):
        if not isinstance(row.get("index"), int):
            raise ValueError(f"{path}:{index}: index must be an integer")
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"{path}:{index}: prompt must be a non-empty string")
        if not isinstance(row.get("response"), str):
            raise ValueError(f"{path}:{index}: response must be a string")
    return rows


def compare_eval_results(
    baseline_path: Path,
    candidate_path: Path,
    output_path: Optional[Path] = None,
) -> dict[str, Any]:
    baseline = load_eval_results(baseline_path)
    candidate = load_eval_results(candidate_path)
    if len(baseline) != len(candidate):
        raise ValueError(
            "eval result lengths differ: "
            f"{baseline_path} has {len(baseline)}, {candidate_path} has {len(candidate)}"
        )

    comparisons: list[dict[str, Any]] = []
    for row_number, (base_row, candidate_row) in enumerate(
        zip(baseline, candidate),
        start=1,
    ):
        if base_row["index"] != candidate_row["index"]:
            raise ValueError(f"row {row_number}: eval indexes differ")
        if base_row["prompt"] != candidate_row["prompt"]:
            raise ValueError(f"row {row_number}: prompts differ")
        comparisons.append(
            {
                "index": base_row["index"],
                "prompt": base_row["prompt"],
                "baseline_response": base_row["response"],
                "candidate_response": candidate_row["response"],
                "baseline_response_chars": len(base_row["response"]),
                "candidate_response_chars": len(candidate_row["response"]),
            }
        )

    report = {
        "baseline_results": str(baseline_path),
        "candidate_results": str(candidate_path),
        "prompt_count": len(comparisons),
        "comparisons": comparisons,
    }
    if output_path is not None:
        resolved_output = output_path.resolve()
        input_paths = {baseline_path.resolve(), candidate_path.resolve()}
        if resolved_output in input_paths:
            raise FileExistsError(
                "comparison output path must not match baseline or candidate eval results"
            )
        if output_path.exists():
            raise FileExistsError(
                f"comparison output path already exists at {output_path}; "
                "choose a new --output path"
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report["output"] = str(output_path)
    return report


def run_eval(request: EvalRequest, dry_run: bool = False) -> dict[str, Any]:
    run_id = request.run_id or utc_eval_run_id()
    run_dir = request.output_root / run_id
    results_path = run_dir / "results.jsonl"
    metadata_path = run_dir / "metadata.json"
    existing_paths = [path for path in (run_dir, results_path, metadata_path) if path.exists()]
    if existing_paths:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"eval run_id {run_id!r} already exists at {existing}; "
            "choose a unique --run-id before running eval"
        )

    prompts = load_eval_prompts(request.prompts_path)
    run_dir.mkdir(parents=True, exist_ok=False)

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

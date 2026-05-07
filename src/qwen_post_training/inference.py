"""Local inference helpers around MLX-LM generation."""

from __future__ import annotations

import importlib.util
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence


DEFAULT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    model: str = DEFAULT_MODEL
    adapter_path: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.0
    seed: Optional[int] = None
    backend: str = "mlx"


def resolve_mlx_command(module_name: str) -> Optional[List[str]]:
    executable = shutil.which(module_name)
    if executable:
        return [executable]
    if importlib.util.find_spec("mlx_lm") is not None:
        return [sys.executable, "-m", module_name]
    return None


def build_generate_command(request: GenerationRequest) -> List[str]:
    base = resolve_mlx_command("mlx_lm.generate") or ["mlx_lm.generate"]
    command = [
        *base,
        "--model",
        request.model,
        "--prompt",
        request.prompt,
        "--max-tokens",
        str(request.max_tokens),
        "--temp",
        str(request.temperature),
    ]
    if request.adapter_path:
        command.extend(["--adapter-path", request.adapter_path])
    if request.seed is not None:
        command.extend(["--seed", str(request.seed)])
    return command


def shell_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def generate(request: GenerationRequest, dry_run: bool = False) -> str:
    if request.backend == "mock":
        return f"mock response for: {request.prompt}"
    if request.backend != "mlx":
        raise ValueError(f"unsupported backend: {request.backend}")

    command = build_generate_command(request)
    if dry_run:
        return shell_command(command)

    if resolve_mlx_command("mlx_lm.generate") is None:
        raise RuntimeError(
            "MLX-LM generation is not installed. Install with: "
            'pip install "mlx-lm[train]"'
        )

    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(detail or f"MLX-LM exited with {completed.returncode}")

    output = completed.stdout.strip()
    if not output:
        output = completed.stderr.strip()
    return output

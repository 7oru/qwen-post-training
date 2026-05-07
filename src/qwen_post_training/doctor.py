"""Environment checks for local Qwen post-training."""

from __future__ import annotations

import importlib.util
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str
    required_for_phase1: bool = False


def command_status(command: str, required_for_phase1: bool = False) -> CheckResult:
    path = shutil.which(command)
    if path is None:
        for python_path in (Path(sys.executable), Path(sys.executable).resolve()):
            sibling = python_path.parent / command
            if sibling.exists():
                path = str(sibling)
                break
    if path is None:
        local_bin = Path.home() / ".local" / "bin" / command
        if local_bin.exists():
            path = str(local_bin)
    return CheckResult(
        name=command,
        status="ok" if path else "missing",
        detail=path or "not found on PATH",
        required_for_phase1=required_for_phase1,
    )


def module_status(module_name: str, required_for_phase1: bool = False) -> CheckResult:
    found = importlib.util.find_spec(module_name) is not None
    return CheckResult(
        name=f"python module {module_name}",
        status="ok" if found else "missing",
        detail="importable" if found else "not importable",
        required_for_phase1=required_for_phase1,
    )


def python_status(version_info: Optional[tuple[int, int]] = None) -> CheckResult:
    major, minor = version_info or sys.version_info[:2]
    ok = (major, minor) >= (3, 11)
    return CheckResult(
        name="python>=3.11",
        status="ok" if ok else "upgrade-needed",
        detail=f"running {major}.{minor}",
        required_for_phase1=True,
    )


def collect_checks() -> List[CheckResult]:
    return [
        python_status(),
        command_status("git", required_for_phase1=True),
        command_status("uv"),
        command_status("mlx_lm.generate", required_for_phase1=True),
        command_status("mlx_lm.server"),
        module_status("mlx_lm", required_for_phase1=True),
        command_status("ollama"),
    ]


def render_report() -> str:
    lines = [
        "qwen-post-training doctor",
        f"python: {sys.version.split()[0]}",
        f"platform: {platform.platform()}",
        f"machine: {platform.machine()}",
        "",
    ]
    for check in collect_checks():
        required = " required-for-phase1" if check.required_for_phase1 else ""
        lines.append(f"{check.name}: {check.status} ({check.detail}){required}")
    lines.extend(
        [
            "",
            "Phase 1 can be dry-run tested without MLX-LM.",
            'Real local Qwen inference requires Python 3.11+ and "mlx-lm[train]".',
        ]
    )
    return "\n".join(lines)


def main() -> int:
    print(render_report())
    return 0

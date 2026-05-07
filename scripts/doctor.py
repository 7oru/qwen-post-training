#!/usr/bin/env python3
"""Check the local skeleton environment."""

from __future__ import annotations

import platform
import shutil
import sys


COMMANDS = [
    "git",
    "uv",
    "mlx_lm.generate",
    "mlx_lm.server",
    "ollama",
]


def main() -> int:
    print("qwen-post-training doctor")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")
    if sys.version_info < (3, 11):
        print("python_status: install Python 3.11+ before training bootstrap")
    else:
        print("python_status: ok")
    print()

    for command in COMMANDS:
        path = shutil.which(command)
        status = path if path else "missing"
        print(f"{command}: {status}")

    print()
    print("Note: missing training commands are expected until bootstrap is built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

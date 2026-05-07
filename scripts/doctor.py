#!/usr/bin/env python3
"""Check the local skeleton environment."""

from __future__ import annotations

import os
import sys


def main() -> int:
    repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    if repo_src not in sys.path:
        sys.path.insert(0, repo_src)

    from qwen_post_training.doctor import main as doctor_main

    return doctor_main()


if __name__ == "__main__":
    raise SystemExit(main())

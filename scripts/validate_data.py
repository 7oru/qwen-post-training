#!/usr/bin/env python3
"""Validate canonical SFT or DPO JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VALID_ROLES = {"system", "user", "assistant"}


def validate_sft(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["messages must be a non-empty list"]

    has_assistant = False
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            errors.append(f"messages[{index}] must be an object")
            continue

        role = message.get("role")
        content = message.get("content")
        if role not in VALID_ROLES:
            errors.append(f"messages[{index}].role must be one of {sorted(VALID_ROLES)}")
        if not isinstance(content, str) or not content.strip():
            errors.append(f"messages[{index}].content must be a non-empty string")
        if role == "assistant":
            has_assistant = True

    if not has_assistant:
        errors.append("messages must include at least one assistant response")
    return errors


def validate_dpo(record: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in ("prompt", "chosen", "rejected"):
        value = record.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} must be a non-empty string")

    if record.get("chosen") == record.get("rejected"):
        errors.append("chosen and rejected must differ")
    return errors


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                yield line_number, None, ["blank lines are not allowed"]
                continue

            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                yield line_number, None, [f"invalid JSON: {exc.msg}"]
                continue

            if not isinstance(value, dict):
                yield line_number, None, ["record must be a JSON object"]
                continue

            yield line_number, value, []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["sft", "dpo"], required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"missing input: {args.input}")
        return 2

    validator = validate_sft if args.kind == "sft" else validate_dpo
    errors_seen = 0
    records_seen = 0

    for line_number, record, parse_errors in iter_jsonl(args.input):
        records_seen += 1
        errors = parse_errors if parse_errors else validator(record or {})
        for error in errors:
            errors_seen += 1
            print(f"{args.input}:{line_number}: {error}")
            if errors_seen >= args.max_errors:
                print(f"stopping after {args.max_errors} errors")
                return 1

    if errors_seen:
        print(f"invalid: {records_seen} records checked, {errors_seen} errors")
        return 1

    print(f"valid: {records_seen} {args.kind} records checked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

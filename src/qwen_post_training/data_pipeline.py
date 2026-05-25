"""Interview-led SFT dataset brief and generation pipeline."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence

from qwen_post_training.inference import DEFAULT_MODEL, GenerationRequest, generate


SFT_INTERVIEW_SKILL = "skills/sft-dataset-interviewer/SKILL.md"
DEFAULT_SPLIT = {"train": 0.8, "validation": 0.1, "test": 0.1}
VALID_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class DatasetBrief:
    slug: str
    goal: str
    audience: str
    assistant_role: str
    task_mix: List[str]
    conversation_patterns: str
    style_rules: str
    source_material: str
    exclusions: str
    failure_modes: str
    eval_prompts: List[str]
    target_examples: int = 1000
    smoke_examples: int = 100
    split: dict[str, float] | None = None

    def spec(self) -> dict[str, Any]:
        return {
            "slug": self.slug,
            "goal": self.goal,
            "audience": self.audience,
            "assistant_role": self.assistant_role,
            "task_mix": self.task_mix,
            "conversation_patterns": self.conversation_patterns,
            "style_rules": self.style_rules,
            "source_material": self.source_material,
            "exclusions": self.exclusions,
            "failure_modes": self.failure_modes,
            "eval_prompts": self.eval_prompts,
            "generation_target": {
                "target_examples": self.target_examples,
                "smoke_examples": self.smoke_examples,
                "split": self.split or DEFAULT_SPLIT,
            },
            "canonical_sft_jsonl": {
                "messages": [
                    {"role": "system", "content": "..."},
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."},
                ]
            },
        }


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "dataset"


def split_csv(values: Optional[str], fallback: Sequence[str]) -> List[str]:
    if not values:
        return list(fallback)
    parts = [part.strip() for part in values.split(",")]
    return [part for part in parts if part]


def prompt_if_missing(label: str, current: Optional[str], fallback: str) -> str:
    if current:
        return current
    try:
        value = input(f"{label} [{fallback}]: ").strip()
    except EOFError:
        value = ""
    return value or fallback


def build_brief_markdown(brief: DatasetBrief) -> str:
    task_lines = "\n".join(f"- {task}" for task in brief.task_mix)
    eval_lines = "\n".join(f"- {prompt}" for prompt in brief.eval_prompts)
    spec_json = json.dumps(brief.spec(), indent=2, ensure_ascii=False)
    return f"""# SFT Dataset Brief: {brief.slug}

Interview workflow: `{SFT_INTERVIEW_SKILL}`

## Goal

{brief.goal}

## Audience

{brief.audience}

## Assistant Role

{brief.assistant_role}

## Task Mix

{task_lines}

## Conversation Patterns

{brief.conversation_patterns}

## Style Rules

{brief.style_rules}

## Source Material

{brief.source_material}

## Exclusions

{brief.exclusions}

## Failure Modes

{brief.failure_modes}

## Eval Prompts

{eval_lines}

## Synthetic Data Prompt Spec

```json
{spec_json}
```
"""


def write_brief(brief: DatasetBrief, brief_dir: Path) -> Path:
    brief_dir.mkdir(parents=True, exist_ok=True)
    path = brief_dir / f"{brief.slug}.md"
    if path.exists():
        raise FileExistsError(
            f"dataset brief slug {brief.slug!r} already exists at {path}; "
            "choose a unique slug before writing the brief"
        )
    path.write_text(build_brief_markdown(brief), encoding="utf-8")
    return path


def extract_json_block(markdown: str) -> dict[str, Any]:
    match = re.search(r"```json\s*(\{.*?\})\s*```", markdown, flags=re.DOTALL)
    if not match:
        raise ValueError("dataset brief does not contain a JSON prompt spec")
    return json.loads(match.group(1))


def load_brief(path: Path) -> DatasetBrief:
    spec = extract_json_block(path.read_text(encoding="utf-8"))
    target = spec.get("generation_target", {})
    return DatasetBrief(
        slug=slugify(str(spec["slug"])),
        goal=str(spec["goal"]),
        audience=str(spec["audience"]),
        assistant_role=str(spec["assistant_role"]),
        task_mix=[str(item) for item in spec.get("task_mix", [])],
        conversation_patterns=str(spec.get("conversation_patterns", "")),
        style_rules=str(spec.get("style_rules", "")),
        source_material=str(spec.get("source_material", "")),
        exclusions=str(spec.get("exclusions", "")),
        failure_modes=str(spec.get("failure_modes", "")),
        eval_prompts=[str(item) for item in spec.get("eval_prompts", [])],
        target_examples=int(target.get("target_examples", 1000)),
        smoke_examples=int(target.get("smoke_examples", 100)),
        split=dict(target.get("split", DEFAULT_SPLIT)),
    )


def validate_sft_record(record: dict[str, Any]) -> List[str]:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return ["messages must be a non-empty list"]

    errors: List[str] = []
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


def user_prompt_key(record: dict[str, Any]) -> str:
    prompts = [
        str(message.get("content", "")).strip().lower()
        for message in record.get("messages", [])
        if isinstance(message, dict) and message.get("role") == "user"
    ]
    return "\n".join(prompts)


def canonical_json(record: dict[str, Any]) -> str:
    return json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class MockTeacher:
    provider_name = "mock"

    def generate_record(self, brief: DatasetBrief, index: int) -> dict[str, Any]:
        task = brief.task_mix[index % len(brief.task_mix)]
        prompt = (
            f"{task} request {index + 1}: help a {brief.audience} with "
            f"{brief.slug} scenario {index + 1}."
        )
        answer = (
            f"For scenario {index + 1}, act as {brief.assistant_role}. "
            f"Address the {task} need directly, follow these style rules: "
            f"{brief.style_rules}. Avoid: {brief.exclusions}."
        )
        return {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"You are {brief.assistant_role}. Use the project goal: "
                        f"{brief.goal}"
                    ),
                },
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": answer},
            ]
        }


class MlxTeacher:
    provider_name = "mlx"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate_record(self, brief: DatasetBrief, index: int) -> dict[str, Any]:
        task = brief.task_mix[index % len(brief.task_mix)]
        teacher_prompt = (
            "Generate exactly one canonical SFT JSON object. Return JSON only.\n"
            f"Dataset goal: {brief.goal}\n"
            f"Audience: {brief.audience}\n"
            f"Assistant role: {brief.assistant_role}\n"
            f"Task: {task}\n"
            f"Style rules: {brief.style_rules}\n"
            f"Exclusions: {brief.exclusions}\n"
            "Shape: {\"messages\":[{\"role\":\"system\",\"content\":\"...\"},"
            "{\"role\":\"user\",\"content\":\"...\"},"
            "{\"role\":\"assistant\",\"content\":\"...\"}]}\n"
            f"Make the user request unique for example {index + 1}."
        )
        output = generate(
            GenerationRequest(
                prompt=teacher_prompt,
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        )
        return parse_json_object(output)


def parse_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("teacher output did not contain a JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("teacher output JSON must be an object")
    return parsed


def split_records(
    records: Sequence[dict[str, Any]],
    split: dict[str, float],
    seed: int,
) -> dict[str, List[dict[str, Any]]]:
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_count = int(total * split.get("train", 0.8))
    if total > 0 and train_count == 0:
        train_count = 1
    validation_count = int(total * split.get("validation", 0.1))
    test_count = total - train_count - validation_count
    return {
        "train": shuffled[:train_count],
        "validation": shuffled[train_count : train_count + validation_count],
        "test": shuffled[train_count + validation_count : train_count + validation_count + test_count],
    }


def validate_sft_splits(splits: dict[str, Sequence[dict[str, Any]]]) -> List[str]:
    errors: List[str] = []
    exact_seen: set[str] = set()
    prompt_owner: dict[str, str] = {}
    total_records = 0

    for split_name, records in splits.items():
        if split_name == "train" and len(records) == 0:
            errors.append("SFT train split contains no records")
        total_records += len(records)
        for index, record in enumerate(records, start=1):
            for error in validate_sft_record(record):
                errors.append(f"{split_name}:{index}: {error}")

            encoded = canonical_json(record)
            if encoded in exact_seen:
                errors.append(f"{split_name}:{index}: duplicate exact record")
            exact_seen.add(encoded)

            prompt_key = user_prompt_key(record)
            if not prompt_key:
                errors.append(f"{split_name}:{index}: missing user prompt")
            elif prompt_key in prompt_owner and prompt_owner[prompt_key] != split_name:
                errors.append(
                    f"{split_name}:{index}: prompt leakage from {prompt_owner[prompt_key]}"
                )
            else:
                prompt_owner[prompt_key] = split_name

    if total_records == 0:
        errors.append("SFT splits contain no records")

    return errors


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> List[dict[str, Any]]:
    records: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object record")
                records.append(value)
    return records


def provider_from_name(
    name: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.2,
):
    if name == "mock":
        return MockTeacher()
    if name == "mlx":
        return MlxTeacher(model=model, max_tokens=max_tokens, temperature=temperature)
    raise ValueError(f"unsupported provider: {name}")


def generate_dataset(
    brief: DatasetBrief,
    provider_name: str,
    count: int,
    processed_root: Path,
    raw_root: Path,
    eval_root: Path,
    seed: int = 7,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.2,
) -> dict[str, Any]:
    provider = provider_from_name(provider_name, model, max_tokens, temperature)
    dataset_dir = processed_root / brief.slug
    raw_dir = raw_root / brief.slug
    eval_path = eval_root / f"{brief.slug}.jsonl"
    existing_paths = [path for path in (dataset_dir, raw_dir, eval_path) if path.exists()]
    if existing_paths:
        existing = ", ".join(str(path) for path in existing_paths)
        raise FileExistsError(
            f"dataset slug {brief.slug!r} already exists at {existing}; "
            "choose a unique slug before generating data"
        )

    candidates: List[dict[str, Any]] = []
    accepted: List[dict[str, Any]] = []

    for index in range(count):
        record = provider.generate_record(brief, index)
        candidates.append(
            {
                "provider": provider.provider_name,
                "brief_slug": brief.slug,
                "index": index + 1,
                "record": record,
            }
        )
        errors = validate_sft_record(record)
        if errors:
            raise ValueError(f"generated record {index + 1} failed validation: {errors}")
        accepted.append(record)

    split = split_records(accepted, brief.split or DEFAULT_SPLIT, seed=seed)
    split_errors = validate_sft_splits(split)
    if split_errors:
        raise ValueError("generated splits failed validation: " + "; ".join(split_errors))

    eval_root.mkdir(parents=True, exist_ok=True)

    write_jsonl(raw_dir / "candidates.jsonl", candidates)
    for split_name, records in split.items():
        write_jsonl(dataset_dir / f"{split_name}.jsonl", records)

    eval_records = [{"prompt": prompt} for prompt in brief.eval_prompts]
    write_jsonl(eval_root / f"{brief.slug}.jsonl", eval_records)

    manifest = {
        "slug": brief.slug,
        "provider": provider.provider_name,
        "count": count,
        "splits": {name: len(records) for name, records in split.items()},
        "processed_dir": str(dataset_dir),
        "raw_candidates": str(raw_dir / "candidates.jsonl"),
        "eval_prompts": str(eval_root / f"{brief.slug}.jsonl"),
        "quality_checks": [
            "canonical_sft_schema",
            "duplicate_exact_records",
            "prompt_leakage_across_splits",
        ],
    }
    (dataset_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_split_dir(dataset_dir: Path) -> List[str]:
    splits = {
        "train": read_jsonl(dataset_dir / "train.jsonl"),
        "validation": read_jsonl(dataset_dir / "validation.jsonl"),
        "test": read_jsonl(dataset_dir / "test.jsonl"),
    }
    return validate_sft_splits(splits)

"""Project CLI."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qwen_post_training.data_pipeline import (
    DatasetBrief,
    generate_dataset,
    load_brief,
    prompt_if_missing,
    slugify,
    split_csv,
    validate_split_dir,
    write_brief,
)
from qwen_post_training.eval import EvalRequest, compare_eval_results, run_eval
from qwen_post_training.inference import DEFAULT_MODEL, GenerationRequest, generate
from qwen_post_training.training import SftTrainingRequest, run_sft_training


def chat(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt).strip()
    if not prompt:
        return interactive_chat(args)

    request = GenerationRequest(
        prompt=prompt,
        model=args.model,
        adapter_path=args.adapter,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        backend=args.backend,
    )
    try:
        print(generate(request, dry_run=args.dry_run))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def interactive_chat(args: argparse.Namespace) -> int:
    print("Starting local chat. Press Ctrl-D to exit.")
    while True:
        try:
            prompt = input("> ").strip()
        except EOFError:
            print()
            return 0
        if not prompt:
            continue
        request = GenerationRequest(
            prompt=prompt,
            model=args.model,
            adapter_path=args.adapter,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            seed=args.seed,
            backend=args.backend,
        )
        try:
            print(generate(request, dry_run=args.dry_run))
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1


def doctor(_: argparse.Namespace) -> int:
    from qwen_post_training.doctor import main as doctor_main

    return doctor_main()


def serve(args: argparse.Namespace) -> int:
    print(f"TODO: serve local HTTP API on {args.host}:{args.port}")
    return 0


def data_brief(args: argparse.Namespace) -> int:
    domain = prompt_if_missing(
        "Domain or product area",
        args.domain,
        "local Qwen post-training workflows",
    ) if args.interactive else args.domain or "local Qwen post-training workflows"
    slug = slugify(args.slug or domain)
    goal = prompt_if_missing(
        "Goal",
        args.goal,
        f"Teach the model to help with {domain}.",
    ) if args.interactive else args.goal or f"Teach the model to help with {domain}."
    audience = prompt_if_missing(
        "Audience",
        args.audience,
        "the local project owner",
    ) if args.interactive else args.audience or "the local project owner"
    assistant_role = prompt_if_missing(
        "Assistant role",
        args.assistant_role,
        f"a concise {domain} assistant",
    ) if args.interactive else args.assistant_role or f"a concise {domain} assistant"
    conversation_patterns = prompt_if_missing(
        "Conversation patterns",
        args.conversation_patterns,
        "Users ask direct task questions and expect practical, structured answers.",
    ) if args.interactive else (
        args.conversation_patterns
        or "Users ask direct task questions and expect practical, structured answers."
    )
    style_rules = prompt_if_missing(
        "Style rules",
        args.style_rules,
        "Be accurate, concise, concrete, and locally actionable.",
    ) if args.interactive else (
        args.style_rules or "Be accurate, concise, concrete, and locally actionable."
    )
    source_material = prompt_if_missing(
        "Source material",
        args.source_material,
        "Use user-provided notes, examples, and project docs.",
    ) if args.interactive else (
        args.source_material or "Use user-provided notes, examples, and project docs."
    )
    exclusions = prompt_if_missing(
        "Exclusions",
        args.exclusions,
        "Do not invent private facts, credentials, or unsupported claims.",
    ) if args.interactive else (
        args.exclusions or "Do not invent private facts, credentials, or unsupported claims."
    )
    failure_modes = prompt_if_missing(
        "Failure modes",
        args.failure_modes,
        "Handle vague prompts by asking for missing details.",
    ) if args.interactive else (
        args.failure_modes or "Handle vague prompts by asking for missing details."
    )

    brief = DatasetBrief(
        slug=slug,
        goal=goal,
        audience=audience,
        assistant_role=assistant_role,
        task_mix=args.task
        or split_csv(args.tasks, ["answer questions", "draft outputs", "critique outputs"]),
        conversation_patterns=conversation_patterns,
        style_rules=style_rules,
        source_material=source_material,
        exclusions=exclusions,
        failure_modes=failure_modes,
        eval_prompts=args.eval_prompt
        or [
            f"Help me with a realistic {domain} task.",
            f"Review this {domain} answer and improve it.",
        ],
        target_examples=args.target_examples,
        smoke_examples=args.smoke_examples,
    )
    path = write_brief(brief, args.brief_dir)
    print(path)
    return 0


def data_generate(args: argparse.Namespace) -> int:
    brief = load_brief(args.brief)
    count = args.count if args.count is not None else brief.target_examples
    if args.smoke:
        count = brief.smoke_examples
    manifest = generate_dataset(
        brief=brief,
        provider_name=args.provider,
        count=count,
        processed_root=args.processed_root,
        raw_root=args.raw_root,
        eval_root=args.eval_root,
        seed=args.seed,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    print(json.dumps(manifest, indent=2))
    return 0


def data_validate_splits(args: argparse.Namespace) -> int:
    errors = validate_split_dir(args.dataset_dir)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(f"valid SFT splits: {args.dataset_dir}")
    return 0


def train_sft(args: argparse.Namespace) -> int:
    request = SftTrainingRequest(
        dataset_dir=args.dataset,
        model=args.model,
        run_id=args.run_id,
        adapters_root=args.adapters_root,
        runs_root=args.runs_root,
        config_path=args.config,
        profile=args.profile,
        iters=args.iters,
        seed=args.seed,
        smoke=args.smoke,
    )
    try:
        result = run_sft_training(request, dry_run=args.dry_run)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def eval_model(args: argparse.Namespace) -> int:
    if args.prompts is None:
        print("error: --prompts is required", file=sys.stderr)
        return 1
    request = EvalRequest(
        prompts_path=args.prompts,
        run_id=args.run_id,
        output_root=args.output_root,
        model=args.model,
        adapter_path=args.adapter,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        backend=args.backend,
    )
    try:
        result = run_eval(request, dry_run=args.dry_run)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def eval_compare(args: argparse.Namespace) -> int:
    try:
        result = compare_eval_results(
            baseline_path=args.baseline,
            candidate_path=args.candidate,
            output_path=args.output,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwenpt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser("doctor", help="Check local tooling")
    doctor_parser.set_defaults(func=doctor)

    chat_parser = subparsers.add_parser("chat", help="Run local adapter chat")
    chat_parser.add_argument("prompt", nargs="*", help="One-shot prompt")
    chat_parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model")
    chat_parser.add_argument("--adapter", help="Adapter directory")
    chat_parser.add_argument("--max-tokens", default=512, type=int)
    chat_parser.add_argument("--temperature", "--temp", default=0.0, type=float)
    chat_parser.add_argument("--seed", type=int)
    chat_parser.add_argument("--backend", choices=["mlx", "mock"], default="mlx")
    chat_parser.add_argument("--dry-run", action="store_true")
    chat_parser.set_defaults(func=chat)

    serve_parser = subparsers.add_parser("serve", help="Run local HTTP server")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", default=8080, type=int)
    serve_parser.set_defaults(func=serve)

    data_parser = subparsers.add_parser("data", help="Dataset brief and SFT data tools")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)

    brief_parser = data_subparsers.add_parser("brief", help="Create an SFT dataset brief")
    brief_parser.add_argument("--slug")
    brief_parser.add_argument("--domain")
    brief_parser.add_argument("--goal")
    brief_parser.add_argument("--audience")
    brief_parser.add_argument("--assistant-role")
    brief_parser.add_argument("--tasks", help="Comma-separated task list")
    brief_parser.add_argument("--task", action="append", help="Task item; may repeat")
    brief_parser.add_argument("--conversation-patterns")
    brief_parser.add_argument("--style-rules")
    brief_parser.add_argument("--source-material")
    brief_parser.add_argument("--exclusions")
    brief_parser.add_argument("--failure-modes")
    brief_parser.add_argument("--eval-prompt", action="append")
    brief_parser.add_argument("--target-examples", default=1000, type=int)
    brief_parser.add_argument("--smoke-examples", default=100, type=int)
    brief_parser.add_argument("--brief-dir", default=Path("dataset_briefs"), type=Path)
    brief_parser.add_argument("--interactive", action="store_true")
    brief_parser.set_defaults(func=data_brief)

    generate_parser = data_subparsers.add_parser(
        "generate", help="Generate canonical SFT JSONL from a dataset brief"
    )
    generate_parser.add_argument("--brief", required=True, type=Path)
    generate_parser.add_argument("--provider", choices=["mock", "mlx"], default="mock")
    generate_parser.add_argument("--count", type=int)
    generate_parser.add_argument("--smoke", action="store_true")
    generate_parser.add_argument("--processed-root", default=Path("data/processed/sft"), type=Path)
    generate_parser.add_argument("--raw-root", default=Path("data/generated"), type=Path)
    generate_parser.add_argument("--eval-root", default=Path("data/eval"), type=Path)
    generate_parser.add_argument("--seed", default=7, type=int)
    generate_parser.add_argument("--model", default=DEFAULT_MODEL)
    generate_parser.add_argument("--max-tokens", default=512, type=int)
    generate_parser.add_argument("--temperature", "--temp", default=0.2, type=float)
    generate_parser.set_defaults(func=data_generate)

    validate_parser = data_subparsers.add_parser(
        "validate-splits", help="Validate train/validation/test SFT JSONL splits"
    )
    validate_parser.add_argument("dataset_dir", type=Path)
    validate_parser.set_defaults(func=data_validate_splits)

    train_parser = subparsers.add_parser("train", help="Local training commands")
    train_subparsers = train_parser.add_subparsers(dest="train_command", required=True)

    sft_parser = train_subparsers.add_parser("sft", help="Run MLX-LM LoRA SFT")
    sft_parser.add_argument(
        "--dataset",
        required=True,
        type=Path,
        help="Directory containing train.jsonl, validation.jsonl, and test.jsonl",
    )
    sft_parser.add_argument("--model", default=DEFAULT_MODEL)
    sft_parser.add_argument("--run-id", help="Stable run id; defaults to UTC timestamp")
    sft_parser.add_argument("--adapters-root", default=Path("adapters/sft"), type=Path)
    sft_parser.add_argument("--runs-root", default=Path("runs/sft"), type=Path)
    sft_parser.add_argument("--config", default=Path("configs/sft.yaml"), type=Path)
    sft_parser.add_argument(
        "--profile",
        help="SFT profile from the config; defaults to local_16gb, or smoke with --smoke",
    )
    sft_parser.add_argument("--iters", type=int, help="Override training iterations")
    sft_parser.add_argument("--seed", default=7, type=int)
    sft_parser.add_argument("--smoke", action="store_true", help="Use smoke-test iteration limits")
    sft_parser.add_argument("--dry-run", action="store_true")
    sft_parser.set_defaults(func=train_sft)

    eval_parser = subparsers.add_parser("eval", help="Run recorded prompt evals")
    eval_parser.add_argument("--prompts", type=Path)
    eval_parser.add_argument("--run-id", help="Stable eval run id; defaults to UTC timestamp")
    eval_parser.add_argument("--output-root", default=Path("runs/eval"), type=Path)
    eval_parser.add_argument("--model", default=DEFAULT_MODEL)
    eval_parser.add_argument("--adapter", help="Adapter directory")
    eval_parser.add_argument("--max-tokens", default=256, type=int)
    eval_parser.add_argument("--temperature", "--temp", default=0.0, type=float)
    eval_parser.add_argument("--seed", default=7, type=int)
    eval_parser.add_argument("--backend", choices=["mlx", "mock"], default="mlx")
    eval_parser.add_argument("--dry-run", action="store_true")
    eval_parser.set_defaults(func=eval_model)
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command")
    compare_parser = eval_subparsers.add_parser(
        "compare",
        help="Compare two eval results JSONL files",
    )
    compare_parser.add_argument("--baseline", required=True, type=Path)
    compare_parser.add_argument("--candidate", required=True, type=Path)
    compare_parser.add_argument("--output", type=Path)
    compare_parser.set_defaults(func=eval_compare)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

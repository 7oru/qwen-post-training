"""Project CLI."""

from __future__ import annotations

import argparse
import sys

from qwen_post_training.inference import DEFAULT_MODEL, GenerationRequest, generate


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

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

"""Project CLI skeleton."""

from __future__ import annotations

import argparse


def chat(args: argparse.Namespace) -> int:
    adapter = args.adapter or "adapters/sft"
    prompt = " ".join(args.prompt).strip()
    if prompt:
        print(f"TODO: run local chat with adapter={adapter!r} prompt={prompt!r}")
    else:
        print(f"TODO: start interactive local chat with adapter={adapter!r}")
    return 0


def serve(args: argparse.Namespace) -> int:
    print(f"TODO: serve local HTTP API on {args.host}:{args.port}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwenpt")
    subparsers = parser.add_subparsers(dest="command", required=True)

    chat_parser = subparsers.add_parser("chat", help="Run local adapter chat")
    chat_parser.add_argument("prompt", nargs="*", help="One-shot prompt")
    chat_parser.add_argument("--adapter", help="Adapter directory")
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

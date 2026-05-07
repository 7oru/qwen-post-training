from __future__ import annotations

import contextlib
import io
import unittest

from qwen_post_training import cli
from qwen_post_training.doctor import python_status
from qwen_post_training.inference import (
    DEFAULT_MODEL,
    GenerationRequest,
    build_generate_command,
    generate,
)


class Phase1CliTests(unittest.TestCase):
    def test_generate_command_uses_default_model(self) -> None:
        command = build_generate_command(GenerationRequest(prompt="hello"))

        self.assertIn("--model", command)
        self.assertIn(DEFAULT_MODEL, command)
        self.assertIn("--prompt", command)
        self.assertIn("hello", command)

    def test_generate_command_includes_adapter_and_seed(self) -> None:
        command = build_generate_command(
            GenerationRequest(
                prompt="hello",
                adapter_path="adapters/sft/run-1",
                max_tokens=128,
                temperature=0.2,
                seed=7,
            )
        )

        self.assertIn("--adapter-path", command)
        self.assertIn("adapters/sft/run-1", command)
        self.assertIn("--seed", command)
        self.assertIn("7", command)
        self.assertIn("128", command)
        self.assertIn("0.2", command)

    def test_mock_backend_returns_non_empty_response(self) -> None:
        response = generate(GenerationRequest(prompt="hello", backend="mock"))

        self.assertIn("hello", response)

    def test_cli_chat_dry_run_prints_mlx_command(self) -> None:
        parser = cli.build_parser()
        args = parser.parse_args(["chat", "--dry-run", "hello"])

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = args.func(args)

        self.assertEqual(exit_code, 0)
        self.assertIn("mlx_lm.generate", stdout.getvalue())
        self.assertIn("hello", stdout.getvalue())

    def test_python_status_reports_upgrade_needed(self) -> None:
        result = python_status((3, 9))

        self.assertEqual(result.status, "upgrade-needed")
        self.assertTrue(result.required_for_phase1)


if __name__ == "__main__":
    unittest.main()

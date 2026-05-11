from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from qwen_post_training import cli
from qwen_post_training.data_pipeline import write_jsonl
from qwen_post_training.eval import EvalRequest, load_eval_prompts, run_eval


class Phase4EvalTests(unittest.TestCase):
    def test_load_eval_prompts_requires_prompt_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prompts.jsonl"
            write_jsonl(path, [{"prompt": "First prompt"}, {"prompt": "Second prompt"}])

            prompts = load_eval_prompts(path)

        self.assertEqual(prompts, ["First prompt", "Second prompt"])

    def test_mock_eval_writes_results_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts_path = root / "prompts.jsonl"
            write_jsonl(prompts_path, [{"prompt": "Help me evaluate."}])

            result = run_eval(
                EvalRequest(
                    prompts_path=prompts_path,
                    run_id="eval-test",
                    output_root=root / "eval-runs",
                    adapter_path="adapters/sft/example",
                    backend="mock",
                )
            )

            metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))
            rows = [
                json.loads(line)
                for line in Path(result["results"]).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

            self.assertEqual(result["prompt_count"], 1)
            self.assertEqual(metadata["adapter_path"], "adapters/sft/example")
            self.assertEqual(metadata["backend"], "mock")
            self.assertIn("Help me evaluate.", rows[0]["response"])

    def test_cli_eval_mock_prints_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts_path = root / "prompts.jsonl"
            write_jsonl(prompts_path, [{"prompt": "Eval prompt"}])
            parser = cli.build_parser()
            args = parser.parse_args(
                [
                    "eval",
                    "--prompts",
                    str(prompts_path),
                    "--output-root",
                    str(root / "runs"),
                    "--run-id",
                    "cli-eval",
                    "--backend",
                    "mock",
                ]
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = args.func(args)

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["run_id"], "cli-eval")
            self.assertEqual(payload["prompt_count"], 1)


if __name__ == "__main__":
    unittest.main()

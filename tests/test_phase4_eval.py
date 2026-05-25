from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from qwen_post_training import cli
from qwen_post_training.data_pipeline import write_jsonl
from qwen_post_training.eval import (
    EvalRequest,
    compare_eval_results,
    load_eval_prompts,
    run_eval,
)


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

    def test_reusing_eval_run_id_fails_before_overwriting_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts_path = root / "prompts.jsonl"
            write_jsonl(prompts_path, [{"prompt": "First eval prompt"}])
            request = EvalRequest(
                prompts_path=prompts_path,
                run_id="duplicate-eval",
                output_root=root / "eval-runs",
                backend="mock",
            )
            first = run_eval(request)
            results_path = Path(first["results"])
            metadata_path = Path(first["metadata"])
            original_results = results_path.read_text(encoding="utf-8")
            original_metadata = metadata_path.read_text(encoding="utf-8")
            write_jsonl(prompts_path, [{"prompt": "Replacement eval prompt"}])

            with self.assertRaisesRegex(FileExistsError, "unique --run-id"):
                run_eval(request)

            self.assertEqual(results_path.read_text(encoding="utf-8"), original_results)
            self.assertEqual(metadata_path.read_text(encoding="utf-8"), original_metadata)

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

    def test_compare_eval_results_pairs_matching_prompts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.jsonl"
            candidate_path = root / "candidate.jsonl"
            output_path = root / "compare" / "report.json"
            write_jsonl(
                baseline_path,
                [{"index": 1, "prompt": "Same prompt", "response": "baseline"}],
            )
            write_jsonl(
                candidate_path,
                [{"index": 1, "prompt": "Same prompt", "response": "candidate"}],
            )

            report = compare_eval_results(baseline_path, candidate_path, output_path)

            self.assertEqual(report["prompt_count"], 1)
            self.assertEqual(report["comparisons"][0]["baseline_response"], "baseline")
            self.assertEqual(report["comparisons"][0]["candidate_response"], "candidate")
            self.assertTrue(output_path.exists())

    def test_compare_eval_results_rejects_output_matching_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.jsonl"
            candidate_path = root / "candidate.jsonl"
            write_jsonl(
                baseline_path,
                [{"index": 1, "prompt": "Same prompt", "response": "baseline"}],
            )
            write_jsonl(
                candidate_path,
                [{"index": 1, "prompt": "Same prompt", "response": "candidate"}],
            )
            original_baseline = baseline_path.read_text(encoding="utf-8")
            original_candidate = candidate_path.read_text(encoding="utf-8")

            for output_path in (baseline_path, candidate_path):
                with self.subTest(output_path=output_path.name):
                    with self.assertRaisesRegex(FileExistsError, "baseline or candidate"):
                        compare_eval_results(baseline_path, candidate_path, output_path)

                    self.assertEqual(baseline_path.read_text(encoding="utf-8"), original_baseline)
                    self.assertEqual(
                        candidate_path.read_text(encoding="utf-8"),
                        original_candidate,
                    )

    def test_compare_eval_results_rejects_existing_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.jsonl"
            candidate_path = root / "candidate.jsonl"
            output_path = root / "report.json"
            write_jsonl(
                baseline_path,
                [{"index": 1, "prompt": "Same prompt", "response": "baseline"}],
            )
            write_jsonl(
                candidate_path,
                [{"index": 1, "prompt": "Same prompt", "response": "candidate"}],
            )
            output_path.write_text("existing report\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                compare_eval_results(baseline_path, candidate_path, output_path)

            self.assertEqual(output_path.read_text(encoding="utf-8"), "existing report\n")

    def test_compare_eval_results_rejects_prompt_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.jsonl"
            candidate_path = root / "candidate.jsonl"
            write_jsonl(
                baseline_path,
                [{"index": 1, "prompt": "Prompt A", "response": "baseline"}],
            )
            write_jsonl(
                candidate_path,
                [{"index": 1, "prompt": "Prompt B", "response": "candidate"}],
            )

            with self.assertRaisesRegex(ValueError, "prompts differ"):
                compare_eval_results(baseline_path, candidate_path)

    def test_cli_eval_compare_prints_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline_path = root / "baseline.jsonl"
            candidate_path = root / "candidate.jsonl"
            write_jsonl(
                baseline_path,
                [{"index": 1, "prompt": "Eval prompt", "response": "baseline"}],
            )
            write_jsonl(
                candidate_path,
                [{"index": 1, "prompt": "Eval prompt", "response": "candidate"}],
            )
            parser = cli.build_parser()
            args = parser.parse_args(
                [
                    "eval",
                    "compare",
                    "--baseline",
                    str(baseline_path),
                    "--candidate",
                    str(candidate_path),
                ]
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = args.func(args)

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["prompt_count"], 1)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from qwen_post_training import cli
from qwen_post_training.dpo import DpoTrainingRequest, dpo_readiness_report, run_dpo_check


class DpoReadinessTests(unittest.TestCase):
    def test_dpo_readiness_reports_unavailable_backend(self) -> None:
        report = dpo_readiness_report(DpoTrainingRequest(config_path=None))

        self.assertEqual(report["status"], "unavailable")
        self.assertIn("does not expose", report["reason"])
        self.assertEqual(report["training"]["beta"], 0.1)

    def test_run_dpo_check_writes_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_dpo_check(
                DpoTrainingRequest(
                    config_path=None,
                    run_id="dpo-test",
                    runs_root=root / "runs",
                    sft_adapter="adapters/sft/example",
                )
            )

            metadata = json.loads(Path(result["metadata"]).read_text(encoding="utf-8"))

        self.assertEqual(result["run_id"], "dpo-test")
        self.assertEqual(metadata["sft_adapter"], "adapters/sft/example")
        self.assertEqual(metadata["status"], "unavailable")

    def test_cli_train_dpo_prints_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            parser = cli.build_parser()
            args = parser.parse_args(
                [
                    "train",
                    "dpo",
                    "--config",
                    str(root / "missing.yaml"),
                    "--runs-root",
                    str(root / "runs"),
                    "--run-id",
                    "cli-dpo",
                ]
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = args.func(args)

            payload = json.loads(stdout.getvalue())

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["run_id"], "cli-dpo")
        self.assertEqual(payload["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()

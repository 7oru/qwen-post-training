from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from qwen_post_training import cli
from qwen_post_training.data_pipeline import DatasetBrief, generate_dataset
from qwen_post_training.training import (
    SftTrainingRequest,
    build_lora_command,
    parse_mlx_metrics,
    prepare_mlx_sft_data,
    run_sft_training,
)


def make_brief() -> DatasetBrief:
    return DatasetBrief(
        slug="phase3-helper",
        goal="Teach the model to help with local SFT runs.",
        audience="local project builders",
        assistant_role="a careful SFT assistant",
        task_mix=["answer questions", "draft commands", "review outputs"],
        conversation_patterns="Users ask practical training workflow questions.",
        style_rules="Be concise and concrete.",
        source_material="Use local project docs.",
        exclusions="Do not invent credentials.",
        failure_modes="Ask for missing details when needed.",
        eval_prompts=["Help me smoke test SFT."],
    )


def make_dataset(root: Path) -> Path:
    generate_dataset(
        brief=make_brief(),
        provider_name="mock",
        count=20,
        processed_root=root / "processed",
        raw_root=root / "raw",
        eval_root=root / "eval",
        seed=7,
    )
    return root / "processed" / "phase3-helper"


class Phase3SftTrainingTests(unittest.TestCase):
    def test_prepare_mlx_sft_data_renames_validation_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = make_dataset(root)
            target_dir = root / "mlx-data"

            prepare_mlx_sft_data(dataset_dir, target_dir)

            self.assertTrue((target_dir / "train.jsonl").exists())
            self.assertTrue((target_dir / "valid.jsonl").exists())
            self.assertTrue((target_dir / "test.jsonl").exists())

    def test_dry_run_writes_run_metadata_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = make_dataset(root)
            result = run_sft_training(
                SftTrainingRequest(
                    dataset_dir=dataset_dir,
                    run_id="sft-test",
                    adapters_root=root / "adapters",
                    runs_root=root / "runs",
                    config_path=None,
                    smoke=True,
                ),
                dry_run=True,
            )

            metadata_path = Path(result["metadata"])
            config_path = metadata_path.parent / "mlx_lora_config.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            mlx_config = json.loads(config_path.read_text(encoding="utf-8"))

            self.assertIn("mlx_lm.lora", result["command"])
            self.assertEqual(result["training_log"], str(root / "runs" / "sft-test" / "training.log"))
            self.assertEqual(result["metrics"], str(root / "runs" / "sft-test" / "metrics.json"))
            self.assertEqual(metadata["adapter_dir"], str(root / "adapters" / "sft-test"))
            self.assertEqual(metadata["training_log"], str(root / "runs" / "sft-test" / "training.log"))
            self.assertEqual(metadata["metrics"], str(root / "runs" / "sft-test" / "metrics.json"))
            self.assertEqual(mlx_config["data"], str(root / "runs" / "sft-test" / "data"))
            self.assertEqual(mlx_config["lora_parameters"]["rank"], 4)
            self.assertTrue(mlx_config["mask_prompt"])

    def test_parse_mlx_metrics_keeps_final_values(self) -> None:
        log_text = "\n".join(
            [
                "Iter 1: Val loss 5.835, Val took 1.677s",
                "Iter 1: Train loss 6.343, Learning Rate 1.000e-05, It/sec 0.539, Tokens/sec 31.241, Trained Tokens 58, Peak mem 4.731 GB",
                "Iter 10: Val loss 5.757, Val took 1.135s",
                "Iter 10: Train loss 5.977, Learning Rate 1.000e-05, It/sec 1.130, Tokens/sec 66.648, Trained Tokens 581, Peak mem 4.823 GB",
                "Test loss 5.655, Test ppl 285.631.",
            ]
        )

        metrics = parse_mlx_metrics(log_text)

        self.assertEqual(metrics["final_validation"]["iteration"], 10)
        self.assertEqual(metrics["final_validation"]["loss"], 5.757)
        self.assertEqual(metrics["final_train"]["peak_memory_gb"], 4.823)
        self.assertEqual(metrics["test"]["perplexity"], 285.631)

    def test_build_lora_command_uses_config_file(self) -> None:
        command = build_lora_command(Path("runs/sft/run-1/mlx_lora_config.json"))

        self.assertIn("--config", command)
        self.assertIn("runs/sft/run-1/mlx_lora_config.json", command)

    def test_cli_train_sft_dry_run_prints_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = make_dataset(root)
            parser = cli.build_parser()
            args = parser.parse_args(
                [
                    "train",
                    "sft",
                    "--dataset",
                    str(dataset_dir),
                    "--runs-root",
                    str(root / "runs"),
                    "--adapters-root",
                    str(root / "adapters"),
                    "--run-id",
                    "cli-smoke",
                    "--config",
                    str(root / "missing-config.yaml"),
                    "--dry-run",
                    "--smoke",
                ]
            )

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = args.func(args)

            payload = json.loads(stdout.getvalue())
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["run_id"], "cli-smoke")
            self.assertTrue(payload["dry_run"])


if __name__ == "__main__":
    unittest.main()

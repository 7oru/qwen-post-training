from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from qwen_post_training.data_pipeline import (
    DatasetBrief,
    generate_dataset,
    load_brief,
    read_jsonl,
    validate_split_dir,
    validate_sft_splits,
    write_brief,
)


def make_brief(slug: str) -> DatasetBrief:
    return DatasetBrief(
        slug=slug,
        goal=f"Teach the model to help with {slug} workflows.",
        audience="local project builders",
        assistant_role=f"a practical {slug} assistant",
        task_mix=["answer questions", "draft outputs", "critique outputs"],
        conversation_patterns="Users ask concrete questions and expect usable answers.",
        style_rules="Be concise, precise, and locally actionable.",
        source_material="Use project notes and user-provided seeds.",
        exclusions="Do not invent credentials or unsupported facts.",
        failure_modes="Ask for missing details when the request is vague.",
        eval_prompts=[f"Help with {slug}.", f"Improve this {slug} answer."],
        target_examples=1000,
        smoke_examples=100,
    )


class Phase2DataPipelineTests(unittest.TestCase):
    def test_brief_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = write_brief(make_brief("support-agent"), Path(tmp))

            loaded = load_brief(path)

        self.assertEqual(loaded.slug, "support-agent")
        self.assertEqual(loaded.target_examples, 1000)
        self.assertIn("answer questions", loaded.task_mix)

    def test_mock_generation_writes_valid_splits(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = generate_dataset(
                brief=make_brief("support-agent"),
                provider_name="mock",
                count=20,
                processed_root=root / "processed",
                raw_root=root / "raw",
                eval_root=root / "eval",
                seed=3,
            )
            dataset_dir = root / "processed" / "support-agent"

            self.assertEqual(manifest["splits"], {"train": 16, "validation": 2, "test": 2})
            self.assertEqual(validate_split_dir(dataset_dir), [])
            self.assertEqual(len(read_jsonl(dataset_dir / "train.jsonl")), 16)
            self.assertTrue((dataset_dir / "manifest.json").exists())
            self.assertTrue((root / "raw" / "support-agent" / "candidates.jsonl").exists())

    def test_two_distinct_mock_datasets_generate_successfully(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for slug in ("support-agent", "code-reviewer"):
                generate_dataset(
                    brief=make_brief(slug),
                    provider_name="mock",
                    count=30,
                    processed_root=root / "processed",
                    raw_root=root / "raw",
                    eval_root=root / "eval",
                    seed=5,
                )
                self.assertEqual(validate_split_dir(root / "processed" / slug), [])

    def test_reusing_dataset_slug_fails_before_overwriting_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_dataset(
                brief=make_brief("support-agent"),
                provider_name="mock",
                count=20,
                processed_root=root / "processed",
                raw_root=root / "raw",
                eval_root=root / "eval",
                seed=3,
            )
            dataset_dir = root / "processed" / "support-agent"
            raw_path = root / "raw" / "support-agent" / "candidates.jsonl"
            eval_path = root / "eval" / "support-agent.jsonl"
            manifest_path = dataset_dir / "manifest.json"
            original_train = (dataset_dir / "train.jsonl").read_text(encoding="utf-8")
            original_raw = raw_path.read_text(encoding="utf-8")
            original_eval = eval_path.read_text(encoding="utf-8")
            original_manifest = manifest_path.read_text(encoding="utf-8")

            with self.assertRaises(FileExistsError):
                generate_dataset(
                    brief=make_brief("support-agent"),
                    provider_name="mock",
                    count=30,
                    processed_root=root / "processed",
                    raw_root=root / "raw",
                    eval_root=root / "eval",
                    seed=11,
                )

            self.assertEqual((dataset_dir / "train.jsonl").read_text(encoding="utf-8"), original_train)
            self.assertEqual(raw_path.read_text(encoding="utf-8"), original_raw)
            self.assertEqual(eval_path.read_text(encoding="utf-8"), original_eval)
            self.assertEqual(manifest_path.read_text(encoding="utf-8"), original_manifest)

    def test_prompt_leakage_is_rejected(self) -> None:
        record = {
            "messages": [
                {"role": "system", "content": "You help."},
                {"role": "user", "content": "Same prompt"},
                {"role": "assistant", "content": "Answer."},
            ]
        }
        errors = validate_sft_splits(
            {"train": [record], "validation": [record], "test": []}
        )

        self.assertTrue(any("prompt leakage" in error for error in errors))


if __name__ == "__main__":
    unittest.main()

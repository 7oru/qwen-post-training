PYTHON ?= python3
export PYTHONPATH := src

.PHONY: doctor validate-sft validate-dpo sft-smoke dpo-smoke chat serve

doctor:
	$(PYTHON) scripts/doctor.py

validate-sft:
	$(PYTHON) scripts/validate_data.py --kind sft --input data/processed/sft/train.jsonl

validate-dpo:
	$(PYTHON) scripts/validate_data.py --kind dpo --input data/processed/dpo/train.jsonl

sft-smoke:
	@echo "TODO: run MLX SFT smoke training from configs/sft.yaml"

dpo-smoke:
	@echo "TODO: run DPO smoke training from configs/dpo.yaml"

chat:
	$(PYTHON) -m qwen_post_training.cli chat

serve:
	$(PYTHON) -m qwen_post_training.cli serve

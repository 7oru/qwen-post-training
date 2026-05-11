PYTHON ?= python3
PROMPT ?= Hello from local Qwen
CHAT_ARGS ?=
MODEL ?= mlx-community/Qwen2.5-7B-Instruct-4bit
DATASET_SLUG ?= local-qwen-helper
DATASET_COUNT ?= 100
SFT_RUN_ID ?=
SFT_PROFILE ?= local_16gb
SFT_ARGS ?=
EVAL_RUN_ID ?=
EVAL_ARGS ?=
export PYTHONPATH := src

.PHONY: doctor validate-sft validate-dpo data-brief-demo data-generate-demo data-validate-demo sft-smoke sft-local eval-demo dpo-smoke chat chat-dry-run test serve

doctor:
	$(PYTHON) scripts/doctor.py

validate-sft:
	$(PYTHON) scripts/validate_data.py --kind sft --input data/processed/sft/train.jsonl

validate-dpo:
	$(PYTHON) scripts/validate_data.py --kind dpo --input data/processed/dpo/train.jsonl

data-brief-demo:
	$(PYTHON) -m qwen_post_training.cli data brief \
		--slug "$(DATASET_SLUG)" \
		--domain "local Qwen post-training" \
		--task "answer project questions" \
		--task "draft dataset examples" \
		--eval-prompt "Help me create an SFT dataset."

data-generate-demo:
	$(PYTHON) -m qwen_post_training.cli data generate \
		--brief dataset_briefs/$(DATASET_SLUG).md \
		--provider mock \
		--count $(DATASET_COUNT)

data-validate-demo:
	$(PYTHON) -m qwen_post_training.cli data validate-splits data/processed/sft/$(DATASET_SLUG)

sft-smoke:
	$(PYTHON) -m qwen_post_training.cli train sft \
		--dataset data/processed/sft/$(DATASET_SLUG) \
		--smoke \
		$(if $(SFT_RUN_ID),--run-id "$(SFT_RUN_ID)",) \
		$(SFT_ARGS)

sft-local:
	$(PYTHON) -m qwen_post_training.cli train sft \
		--dataset data/processed/sft/$(DATASET_SLUG) \
		--profile "$(SFT_PROFILE)" \
		$(if $(SFT_RUN_ID),--run-id "$(SFT_RUN_ID)",) \
		$(SFT_ARGS)

eval-demo:
	$(PYTHON) -m qwen_post_training.cli eval \
		--prompts data/eval/$(DATASET_SLUG).jsonl \
		$(if $(EVAL_RUN_ID),--run-id "$(EVAL_RUN_ID)",) \
		$(EVAL_ARGS)

dpo-smoke:
	@echo "TODO: run DPO smoke training from configs/dpo.yaml"

chat:
	$(PYTHON) -m qwen_post_training.cli chat --model "$(MODEL)" $(CHAT_ARGS) "$(PROMPT)"

chat-dry-run:
	$(PYTHON) -m qwen_post_training.cli chat --dry-run --model "$(MODEL)" $(CHAT_ARGS) "$(PROMPT)"

test:
	$(PYTHON) -m unittest discover -s tests

serve:
	$(PYTHON) -m qwen_post_training.cli serve

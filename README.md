# qwen-post-training

Local-first post-training toolkit for Qwen2.5 7B on Apple Silicon.

This project is building a private workflow to fine-tune
`Qwen2.5-7B-Instruct` on a Mac mini M4 with 16 GB unified memory using
MLX-friendly adapter training. The target path is:

1. Interview the user to define a custom SFT dataset.
2. Generate or collect about `1000` high-quality training examples.
3. Validate and normalize data into canonical SFT/DPO JSONL.
4. Run SFT locally with LoRA/QLoRA adapters.
5. Run DPO locally when memory allows.
6. Run the fine-tuned adapter from CLI.
7. Serve the same local model over HTTP later.

The default design keeps private data, training, and inference on the local
machine. Cloud LLMs or hosted training can be added later only as explicit
opt-in paths.

## Current Status

This repository is in planning/bootstrap stage.

Implemented documentation artifacts:

- [ROADMAP.md](ROADMAP.md): full local training roadmap, 16 GB constraints,
  synthetic data pipeline, SFT/DPO formats, CLI/HTTP goals, and 64 GB scale
  plan.
- [skills/sft-dataset-interviewer/SKILL.md](skills/sft-dataset-interviewer/SKILL.md):
  repo-local skill for interviewing the user and producing a concrete SFT
  dataset brief before data generation.
- [docs/references/sglang-post-training.md](docs/references/sglang-post-training.md):
  review of SGLang's post-training infrastructure ideas and which ones fit
  this local-first project.

Implementation still to build:

- Environment/bootstrap scripts for Python 3.11+ and MLX-LM.
- Custom data generators and validators.
- Base model inference check.
- SFT and DPO training configs.
- Local HTTP serving wrapper.

Phase 1 CLI scaffolding is available:

```bash
make doctor
make chat-dry-run PROMPT="hello"
PYTHONPATH=src python3 -m qwen_post_training.cli chat --backend mock "hello"
```

Real local Qwen inference requires Python 3.11+ and `mlx-lm[train]`.

## Custom Data Workflow

Custom data is first-class. The project should support both user-provided data
and LLM-generated synthetic data.

The SFT data workflow starts with the dataset interviewer skill:

1. Talk with the user for a few focused rounds.
2. Capture the desired model behavior, audience, task mix, style, boundaries,
   and evaluation prompts.
3. Produce a dataset brief.
4. Generate a first smoke batch of about `100` examples.
5. Review, score, deduplicate, and validate.
6. Scale toward about `1000` examples.
7. Export accepted records to canonical SFT chat JSONL.

Canonical SFT record:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Canonical DPO record:

```json
{"prompt":"...","chosen":"...","rejected":"..."}
```

## Hardware Strategy

On the current 16 GB Mac mini M4, the project should use:

- 4-bit MLX Qwen2.5 7B instruct base model.
- LoRA/QLoRA adapter training only.
- Small batch size, short context, low LoRA rank, and conservative DPO tests.

On a future 64 GB M4/M5-class machine, the project should scale by increasing:

- Context length.
- LoRA rank.
- Trainable layers.
- Validation and preference datasets.
- DPO experiment depth.
- Optional less-aggressive quantization.

Full-weight 7B fine-tuning is not the default local target. The near-term goal
is reliable local adapter training that can grow with larger Apple Silicon
machines.

## Planned Stack

- Base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`.
- Upstream reference: `Qwen/Qwen2.5-7B-Instruct`.
- Training approach: MLX LoRA/QLoRA SFT, then DPO if stable.
- Inference: local CLI first.
- Serving: localhost HTTP API after CLI inference works.

See [ROADMAP.md](ROADMAP.md) for the detailed build plan.

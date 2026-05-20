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

- DPO training command.

Implemented code paths:

- Real local Qwen CLI inference through `qwenpt chat`.
- SFT dataset brief generation through `qwenpt data brief`.
- Deterministic mock SFT data generation and split validation through
  `qwenpt data generate` and `qwenpt data validate-splits`.
- Conservative MLX-LM LoRA SFT command construction, split preparation, and
  per-run metadata through `qwenpt train sft`.

## Local CLI Chat

Use a repo-local virtual environment for real model inference. The project
expects Python 3.11+; Python 3.12 is the tested local setup.

```bash
cd /Users/ricktu/qwen-post-training

curl -LsSf https://astral.sh/uv/install.sh | sh
~/.local/bin/uv python install 3.12
~/.local/bin/uv venv --python 3.12 .venv
~/.local/bin/uv pip install -r requirements.txt
```

Check the environment and run a real local prompt:

```bash
.venv/bin/qwenpt doctor
.venv/bin/qwenpt chat --max-tokens 64 "Say hi in five words."
```

The first real run downloads `mlx-community/Qwen2.5-7B-Instruct-4bit` into the
local Hugging Face cache. On the Mac mini M4 16 GB smoke test, a short prompt
completed successfully with about 4.4 GB peak memory reported by MLX-LM.

Useful test commands:

```bash
make doctor
make chat-dry-run PROMPT="hello"
.venv/bin/qwenpt chat --dry-run "hello"
.venv/bin/qwenpt chat --backend mock "hello"
```

If Python 3.12 is already installed, standard `venv` also works:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

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

Create a dataset brief and generate deterministic mock SFT data:

```bash
.venv/bin/qwenpt data brief \
  --slug local-qwen-helper \
  --domain "local Qwen post-training" \
  --task "answer project questions" \
  --task "draft dataset examples" \
  --eval-prompt "Help me create an SFT dataset."

.venv/bin/qwenpt data generate \
  --brief dataset_briefs/local-qwen-helper.md \
  --provider mock \
  --count 1000

.venv/bin/qwenpt data validate-splits data/processed/sft/local-qwen-helper
```

Use `--provider mlx` to generate through the local Qwen teacher model instead
of the deterministic mock provider. Start with `--smoke` or a small `--count`
before attempting `1000` examples.

Canonical SFT record:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Canonical DPO record:

```json
{"prompt":"...","chosen":"...","rejected":"..."}
```

## Local SFT Smoke Training

After generating and validating an SFT dataset, run a conservative MLX-LM LoRA
smoke training job:

```bash
.venv/bin/qwenpt train sft \
  --dataset data/processed/sft/local-qwen-helper \
  --smoke
```

To inspect the exact `mlx_lm.lora` invocation and generated run metadata
without starting training:

```bash
.venv/bin/qwenpt train sft \
  --dataset data/processed/sft/local-qwen-helper \
  --smoke \
  --dry-run
```

Each run writes MLX-ready split files, metadata, `training.log`, and
`metrics.json` under `runs/sft/<run_id>`, and saves adapters under
`adapters/sft/<run_id>`.

The default non-smoke profile is `local_16gb`, which keeps batch size, context,
rank, and layer count small but trains longer than the smoke path:

```bash
make sft-local PYTHON=.venv/bin/python DATASET_SLUG=local-qwen-helper
```

Use `--profile smoke` or `--profile local_16gb` to choose a profile explicitly.
Use `--iters` for one-off iteration overrides without editing
`configs/sft.yaml`.

Run recorded before/after evals against fixed prompt files:

```bash
.venv/bin/qwenpt eval \
  --prompts data/eval/local-qwen-helper.jsonl \
  --run-id baseline-local-qwen-helper

.venv/bin/qwenpt eval \
  --prompts data/eval/local-qwen-helper.jsonl \
  --adapter adapters/sft/sft-local16-20260511T105810Z \
  --run-id sft-local16-local-qwen-helper
```

Eval runs write `metadata.json` and `results.jsonl` under `runs/eval/<run_id>`.
Compare two recorded eval runs:

```bash
.venv/bin/qwenpt eval compare \
  --baseline runs/eval/eval-baseline-local-qwen-helper-20260511T105810Z/results.jsonl \
  --candidate runs/eval/eval-sft-local16-local-qwen-helper-20260511T105810Z/results.jsonl \
  --output runs/eval/compare-local-qwen-helper.json
```

## Local HTTP Serving

Serve the same local generation path behind an OpenAI-compatible chat endpoint:

```bash
.venv/bin/qwenpt serve \
  --host 127.0.0.1 \
  --port 8080 \
  --adapter adapters/sft/sft-local16-20260511T105810Z
```

Then call it from another terminal:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mlx-community/Qwen2.5-7B-Instruct-4bit","messages":[{"role":"user","content":"Help me create an SFT dataset."}],"max_tokens":128}'
```

The server also exposes `GET /health` and `GET /v1/models`. Use
`--backend mock` for endpoint smoke tests without loading MLX.

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

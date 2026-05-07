# Local Qwen2.5 Post-Training Roadmap

This repository is for local-only post-training of Qwen2.5 7B on Apple
Silicon, starting with a Mac mini M4 with 16 GB unified memory. The first
target is a practical adapter workflow:

1. Supervised fine-tuning (SFT) on custom chat data.
2. Direct Preference Optimization (DPO) on custom preference pairs.
3. Local fine-tuned inference from the command line.
4. Local HTTP serving for future app integration.

The default path keeps private training data on the local machine. No cloud
training or hosted inference service is required.

## V1 Target

Use an MLX-friendly 4-bit Qwen2.5 7B instruct base model and train LoRA/QLoRA
adapters locally. The adapter-first approach keeps the project realistic on
16 GB memory while preserving a clean scale path for larger Apple Silicon
machines later.

Recommended base:

- `mlx-community/Qwen2.5-7B-Instruct-4bit` for local MLX training/inference.
- `Qwen/Qwen2.5-7B-Instruct` as the upstream model reference.

Training order:

1. Validate custom data.
2. Run an SFT smoke test.
3. Run full SFT adapter training.
4. Run a DPO smoke test from the SFT adapter.
5. Run DPO adapter training if memory and quality checks pass.
6. Serve the tuned adapter through CLI first, then HTTP.

## Custom Data Support

Custom data is first-class. The repository should normalize private datasets
into two canonical JSONL formats before training.

### SFT Chat JSONL

Each line is one chat example. The `messages` array should follow chat roles
used by instruct models.

```json
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Write a short welcome email."},{"role":"assistant","content":"Subject: Welcome\n\nHi there, welcome aboard."}]}
```

Minimum validation:

- `messages` is present and non-empty.
- Each message has `role` and `content`.
- Supported roles are `system`, `user`, and `assistant`.
- Each example has at least one assistant response.
- Empty content, duplicate exact records, and train/validation/test leakage are rejected.

### DPO Preference JSONL

Each line is one preference example with a prompt and two candidate answers.

```json
{"prompt":"Explain LoRA in one paragraph.","chosen":"LoRA trains small adapter matrices while keeping the base model frozen, which reduces memory and compute needs.","rejected":"LoRA retrains every model weight from scratch."}
```

Minimum validation:

- `prompt`, `chosen`, and `rejected` are present and non-empty.
- `chosen` and `rejected` are not identical.
- Prompts duplicated across splits are rejected.
- Examples that exceed the configured token budget are reported before training.

Future importers can convert Alpaca, ShareGPT, CSV, or project-specific data
into these canonical SFT and DPO JSONL formats. Training code should consume
only the canonical formats.

### Synthetic Data Generation Pipeline

Custom data can also be generated as a separate local pipeline before training.
The first target is about `1000` high-quality examples produced by an LLM,
then filtered, deduplicated, validated, and exported into the canonical SFT or
DPO JSONL formats.

Before generation, use `skills/sft-dataset-interviewer/SKILL.md` to talk with
the user for a few focused rounds and produce an SFT dataset brief. The brief
should define the target behavior, audience, task mix, style rules, exclusions,
quality checks, and eval prompts that guide synthetic data generation.

Default generation policy:

- Use a local teacher model by default so private topic seeds stay on the
  machine.
- Allow an external teacher LLM only through an explicit opt-in config later.
- Generate from user-provided seed topics, task descriptions, rubrics, source
  notes, or examples.
- Store raw generated records separately from validated training records.
- Keep every generated record traceable to its seed, generator prompt, and
  generation run.
- Review or automatically score generated data before it can enter a training
  split.

Suggested flow:

1. Interview the user and produce an SFT dataset brief.
2. Define a small seed set of domains, intents, tone rules, and failure modes.
3. Generate candidate SFT conversations and DPO preference pairs.
4. Critique each candidate with a rubric or judge prompt.
5. Remove duplicates, low-scoring records, unsafe records, and overlong records.
6. Export accepted records to canonical SFT and DPO JSONL.
7. Split into train/validation/test sets with prompt leakage checks.

For 16 GB local runs, generated data should start small: generate `100` to
`200` examples first, inspect quality, then scale toward `1000`. On a future
64 GB machine, the same pipeline can use larger teacher models, longer source
contexts, and more judge passes before training.

## Current 16 GB Mac Mini Path

The current machine is best suited to conservative adapter training.

Default constraints:

- Use a 4-bit MLX base model.
- Train LoRA/QLoRA adapters only.
- Keep batch size at `1` for smoke tests and early runs.
- Use gradient accumulation instead of large batches.
- Start with context length `512` or `1024`.
- Start with LoRA rank `4` or `8`.
- Train a small number of layers first, then expand only after memory is stable.
- Keep evaluation sets small but fixed so runs remain comparable.

Out of scope on this machine:

- Full-weight Qwen2.5 7B fine-tuning.
- Large-context DPO runs.
- Multi-adapter experiments before the baseline SFT/DPO path is working.

The first implementation should prefer reliability over maximum training
throughput. A slow local run that completes and produces measurable behavior
change is more valuable than an aggressive configuration that repeatedly
crashes from memory pressure.

## SFT Plan

SFT teaches the model the desired domain style, structure, and task behavior.

Initial settings:

- Base model: `mlx-community/Qwen2.5-7B-Instruct-4bit`.
- Data: canonical SFT chat JSONL.
- Adapter type: LoRA/QLoRA.
- Batch size: `1`.
- Gradient accumulation: `8` to `16`.
- Context length: start at `512`; raise to `1024` after a successful smoke run.
- LoRA rank: start at `4`; raise to `8` if memory allows.
- Prompt masking: train primarily on assistant response tokens.

Expected outputs:

- SFT adapter directory.
- Training logs with loss and validation loss.
- A fixed set of before/after eval responses.

## DPO Plan

DPO should start only after SFT inference works locally. It should refine
preferences, refusals, tone, ranking, and answer quality using chosen/rejected
pairs.

Initial settings:

- Start from the SFT adapter.
- Data: canonical DPO preference JSONL.
- Batch size: `1`.
- Context length: start at `512`.
- Beta: start around `0.05` to `0.1`.
- Use short preference examples for the first smoke run.
- Keep the reference behavior stable and compare outputs against the SFT model.

Risk:

DPO is more memory-sensitive than SFT because it compares preferred and
rejected responses. If DPO does not fit reliably on 16 GB, reduce context
length, trainable layers, and LoRA rank before considering a larger machine.

## Local Inference

The first inference target is command-line use with the fine-tuned adapter.

Expected CLI behavior:

- Load the 4-bit base model.
- Load the selected adapter path.
- Accept either a one-shot prompt or an interactive chat session.
- Print model output to stdout.

Example future commands:

```bash
make chat
qwenpt chat "Summarize this training roadmap."
qwenpt chat --adapter adapters/sft-dpo
```

## Local HTTP Serving

HTTP serving should come after CLI inference is stable. The first server can be
thin and local-only.

Expected HTTP behavior:

- Bind to localhost by default.
- Load the same base model and adapter used by CLI inference.
- Expose a chat/completions-style endpoint for future app integration.
- Keep request/response logging local.

Example future commands:

```bash
make serve
curl http://localhost:8080/v1/chat/completions
```

SGLang is a useful reference for the serving contract and future rollout
lifecycle. See `docs/references/sglang-post-training.md` for what to borrow
from SGLang and what to avoid in the 16 GB local-first path.

## 64 GB M4/M5 Scale Plan

A future 64 GB Apple Silicon machine should still use an adapter-first path by
default. The goal is not to promise full-weight 7B training, but to unlock
larger, more stable local experiments.

Scale-up priorities:

- Increase context length from `512`/`1024` toward `2048` or higher.
- Increase LoRA rank from `4`/`8` toward `16` or `32`.
- Train more layers or all adapter-supported layers.
- Use larger validation and preference sets.
- Run more DPO experiments with safer evaluation coverage.
- Test less aggressive quantization if memory allows.
- Evaluate larger local bases only after Qwen2.5 7B adapter training is stable.

64 GB should improve quality, stability, and experimentation speed. Full-weight
fine-tuning remains a future-machine topic and should be documented separately
only after adapter training has reached its limits.

## Milestone Ladder

1. Docs starter commit.
2. Environment/bootstrap scripts.
3. SFT dataset interviewer skill.
4. Synthetic data generation pipeline for about `1000` custom examples.
5. Data validators for SFT and DPO JSONL.
6. Base model CLI inference.
7. SFT smoke run.
8. Full local SFT run.
9. DPO smoke run.
10. Local tuned CLI inference.
11. Local HTTP serving.
12. 64 GB scale-profile configs.

## Acceptance Criteria

- The project can validate custom SFT and DPO datasets before training.
- The project can interview the user and produce a concrete SFT dataset brief.
- The project can generate, filter, and export about `1000` synthetic custom
  examples into the canonical SFT/DPO formats.
- SFT training completes locally on the 16 GB Mac mini using conservative
  settings.
- DPO has a smoke-test path and a documented fallback if memory is insufficient.
- Fine-tuned adapter inference works from CLI.
- The same adapter can later be served over local HTTP.
- The 64 GB plan increases local adapter capacity without making full-weight
  fine-tuning the default promise.

# Qwen Local Post-Training Implementation Milestones

This plan turns the roadmap into testable implementation phases for a local
Mac-first Qwen2.5 post-training loop.

## Summary

Beta proves the core local loop on the 16 GB Mac mini:

1. Run Qwen2.5 locally from CLI.
2. Generate two custom SFT datasets through an interview-led workflow.
3. SFT Qwen locally with conservative MLX LoRA/QLoRA settings.
4. Run the SFT adapter locally from CLI.

DPO stays post-Beta because it is memory-sensitive and not needed to prove the
first complete custom-data-to-adapter loop.

## Beta Milestones

### Phase 1: Local Qwen CLI

- Implement `qwenpt chat` and `make chat` using MLX-LM generation.
- Default model: `mlx-community/Qwen2.5-7B-Instruct-4bit`.
- Support one-shot prompts and adapter-free baseline inference.
- Include a dry-run path so command construction can be tested without model
  downloads.

Acceptance:

- `make doctor` reports Python and local model tooling status.
- `qwenpt chat "hello"` returns non-empty local output when MLX-LM and the
  model are installed.
- `qwenpt chat --dry-run "hello"` prints the MLX command without requiring the
  model.

### Phase 2: Two Custom SFT Datasets

- Add dataset interview commands that produce `dataset_briefs/<slug>.md`.
- Add local-first teacher generation with a deterministic mock provider for
  tests and an MLX local provider for real runs.
- Generate two distinct interviewed dataset briefs.
- For each dataset, generate a `100` example smoke batch, then scale to `1000`
  accepted SFT examples.
- Export train/validation/test JSONL splits.

Acceptance:

- Both datasets pass schema validation.
- Duplicate records are rejected.
- Prompt leakage across splits is rejected.
- Quality-review gates run before data enters training splits.

### Phase 3: Local SFT

Status: initial CLI wrapper implemented and smoke-tested locally. `qwenpt train
sft` prepares MLX-LM split files, writes per-run config/metadata/logs/metrics,
supports dry-run command inspection, and saves adapters under versioned run
directories. Named profiles now separate `smoke` from the default `local_16gb`
run. The first logged smoke run completed as `sft-smoke-logged-20260511T104712Z`
with final validation loss `5.757`, test loss `5.655`, test perplexity
`285.631`, and peak memory `4.823 GB`.

- Implement `qwenpt train sft` and `make sft-smoke` around MLX-LM LoRA
  training.
- Use canonical SFT chat JSONL.
- Start with batch size `1`, short context, low LoRA rank, conservative layer
  count, and prompt masking.
- Output versioned adapters under `adapters/sft/<run_id>`.
- Write run metadata under `runs/sft/<run_id>`.

Acceptance:

- SFT smoke training completes on a tiny subset.
- Full local SFT completes on one generated dataset.
- Adapter files and validation metrics are written.

### Phase 4: Local SFT Adapter Inference

Status: complete for Beta. Adapter inference works through
`qwenpt chat --adapter`, `qwenpt eval` records fixed prompt runs with model,
adapter, seed, sampling params, backend, metadata, and JSONL results under
`runs/eval/<run_id>`, and `qwenpt eval compare` produces deterministic
side-by-side reports for baseline/adapter results.

- Extend `qwenpt chat --adapter adapters/sft/<run_id>` to load trained
  adapters.
- Add `qwenpt eval` for fixed before/after prompts.
- Record model, adapter, seed, sampling params, backend, and prompt set for
  every eval run.

Acceptance:

- Baseline and SFT adapter both answer the same eval prompts.
- Adapter inference works from CLI.
- Eval metadata is reproducible enough to compare runs.

## Future Milestones

### 64 GB Phase 1: Stronger Local Adapters

- Add 64 GB profiles with larger context, higher LoRA rank, more trainable
  layers, larger eval sets, and safer DPO experiments.
- Keep adapter-first as the default.
- Leave full-weight fine-tuning out of scope unless explicitly revisited.

Acceptance:

- 64 GB profile completes SFT smoke, then full SFT.
- Eval coverage is broader than the 16 GB profile.

### Local HTTP Endpoint

Status: implemented for local Beta. `qwenpt serve` starts a local-only
OpenAI-compatible server with `GET /health`, `GET /v1/models`, and
`POST /v1/chat/completions`. Requests route through the same local inference
pipeline used by CLI chat and support the selected base model, backend, seed,
sampling params, and adapter path.

- Add a localhost OpenAI-compatible `POST /v1/chat/completions` endpoint.
- Route requests through the same local inference pipeline used by CLI.
- Keep SGLang as an optional future backend reference, not the default v1
  dependency.

Acceptance:

- `curl http://localhost:8080/v1/chat/completions` returns local model output
  using the selected base model and adapter.

## Test Plan

- Unit tests: validators, split logic, dataset brief parsing, provider
  interfaces, CLI argument parsing.
- Fixture tests: mock teacher generates deterministic tiny SFT datasets without
  requiring a model download.
- Integration tests: MLX baseline generation, SFT smoke training, SFT adapter
  generation.
- Acceptance tests: two generated SFT datasets with `1000` accepted examples
  each; one successful SFT adapter; local CLI inference before and after SFT.
- Regression tests: fixed eval prompts saved before SFT and rerun after SFT
  with metadata.

## Defaults

- Teacher policy: local-first.
- External APIs: explicit opt-in only.
- Dataset success: two distinct interviewed SFT dataset briefs, each producing
  validated train/validation/test JSONL.
- DPO: post-Beta after SFT training and SFT adapter inference work locally.

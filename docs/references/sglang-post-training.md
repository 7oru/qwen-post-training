# SGLang Post-Training Infrastructure Review

Review date: 2026-05-07

SGLang is most useful to this project as a reference for rollout, evaluation,
adapter serving, and HTTP API design. It should not replace the initial
MLX-based SFT/DPO training path for the 16 GB Mac mini.

## What SGLang Provides

SGLang's post-training docs position it as the inference and rollout backend
inside modern RL/post-training systems. The main ideas are engine sleep/wake,
weight refit APIs, pausing and continuing generation, deterministic inference,
and cache-aware load balancing.

Useful sources:

- [Post-Training Integration](https://docs.sglang.io/docs/references/post_training_integration)
- [SGLang for RL Systems](https://docs.sglang.io/docs/advanced_features/sglang_for_rl)
- [LoRA Serving](https://docs.sglang.io/docs/advanced_features/lora)
- [Apple Silicon with Metal](https://docs.sglang.io/docs/hardware-platforms/apple_metal)
- [Deterministic Inference](https://docs.sglang.io/docs/advanced_features/deterministic_inference)
- [SGLang Model Gateway](https://docs.sglang.io/docs/advanced_features/sgl_model_gateway)
- [OpenAI-Compatible APIs](https://docs.sglang.io/docs/basic_usage/openai_api_completions)

## Ideas To Reuse

- **OpenAI-compatible HTTP contract**: expose `/v1/chat/completions` for the
  local server so future clients can swap between MLX-LM, SGLang, or another
  backend with minimal client changes.
- **Adapter naming**: use stable adapter IDs and consider SGLang's
  `base-model:adapter-name` convention when designing HTTP requests.
- **Dynamic adapter lifecycle**: model future endpoints after SGLang's
  load/unload LoRA adapter flow, but implement only after basic single-adapter
  serving works.
- **Deterministic eval runs**: store sampling parameters, seeds, adapter
  version, prompt set, and backend name for every before/after evaluation.
- **Checkpoint/update boundary**: treat trained adapters as immutable versioned
  artifacts. Load from disk between SFT, DPO, eval, and serving steps.
- **Rollout control concepts**: keep pause/resume and weight-refit patterns as
  future references for online RL or GRPO-style work, not for the first DPO
  implementation.
- **Apple Silicon serving option**: SGLang documents an MLX path with
  `SGLANG_USE_MLX=1`, so it can be tested later as an alternate local serving
  backend.

## What Not To Copy For V1

- Do not make SGLang the local trainer. The 16 GB path should remain MLX
  LoRA/QLoRA SFT first, then DPO if memory allows.
- Do not add distributed rollout, NCCL weight update, gateway routing, or
  multi-worker scheduling to the first local implementation.
- Do not depend on CUDA-only deterministic inference behavior for Mac evals.
  Borrow the reproducibility discipline, but verify what the MLX backend
  actually supports.
- Do not optimize for multi-LoRA batching until single-adapter inference and
  HTTP serving are reliable.

## Recommended Use In This Repo

Short term:

1. Keep MLX-LM as the default trainer and first inference backend.
2. Make the local HTTP API OpenAI-compatible.
3. Record eval metadata with backend, adapter, prompt set, sampling params, and
   seed.
4. Add a backend abstraction in the CLI/server only when there are at least two
   runnable local backends.

Medium term:

1. Test SGLang on Apple Silicon after MLX-LM CLI/server inference works.
2. Add an optional `serve_sglang` profile if SGLang can load the selected Qwen
   base and adapter reliably on the target machine.
3. Borrow SGLang's adapter naming and dynamic adapter lifecycle for local HTTP
   serving.

Future scale:

1. Use SGLang Model Gateway ideas only for a 64 GB-plus or multi-machine setup.
2. Revisit deterministic inference and weight update APIs if the project moves
   from offline SFT/DPO into online RL or GRPO-style rollouts.

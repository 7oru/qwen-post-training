# Data Layout

Training data is private by default. The repository tracks this layout but
ignores local dataset contents.

- `seeds/`: user-provided topic seeds, source notes, rubrics, and task ideas.
- `raw/`: original imported datasets before normalization.
- `generated/`: LLM-generated candidate examples before review.
- `processed/sft/`: accepted canonical SFT chat JSONL.
- `processed/dpo/`: accepted canonical DPO preference JSONL.
- `eval/`: fixed prompts for before/after model comparisons.

Canonical SFT record:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Canonical DPO record:

```json
{"prompt":"...","chosen":"...","rejected":"..."}
```

---
name: sft-dataset-interviewer
description: Use when designing a custom SFT dataset before generation or curation. Guides a short multi-round interview with the user, then produces a concrete dataset brief for generating or collecting chat-style SFT examples.
---

# SFT Dataset Interviewer

Use this skill before generating or curating custom SFT data. The goal is to
turn a rough idea into a dataset brief that can drive about `1000` high-quality
chat examples.

Keep the conversation short and decision-oriented. Ask in rounds, not a long
questionnaire. Prefer 3 to 5 questions per round.

## Workflow

### Round 1: Intent

Learn what behavior the fine-tuned model should gain.

Ask for:

- The domain or product area.
- The target users.
- The assistant's desired role.
- The top 3 to 5 task types the model should handle.
- Examples of answers the user likes or dislikes, if available.

### Round 2: Data Shape

Turn the intent into trainable SFT examples.

Ask for:

- Input styles users will actually type.
- Expected answer formats.
- Tone and voice rules.
- Required terminology, constraints, or policies.
- Source notes, docs, FAQs, examples, or seed topics that can guide generation.

### Round 3: Boundaries

Prevent bad or low-value examples from entering the dataset.

Ask for:

- Topics to exclude.
- Claims the model should avoid making.
- Safety, privacy, legal, or brand constraints.
- Failure modes to include as training examples.
- Evaluation prompts that should improve after fine-tuning.

### Optional Round 4: Generation Plan

Use this round only if the dataset target is still vague.

Ask for:

- The desired example count.
- The rough split between task types.
- Whether examples should include system prompts.
- Whether generation must use a local teacher model only.
- Whether a human review pass is required before training.

## Output

After the interview, produce a dataset brief with these sections:

- `Goal`: one paragraph describing the desired model behavior.
- `Audience`: who will use the model and in what context.
- `Assistant role`: how the model should act.
- `Task mix`: task categories with approximate percentages.
- `Conversation patterns`: expected user inputs and assistant outputs.
- `Style rules`: tone, formatting, length, and terminology.
- `Source material`: seed topics, docs, notes, examples, or gaps.
- `Exclusions`: topics, claims, and behaviors to avoid.
- `Failure modes`: mistakes the dataset should teach the model to handle.
- `Generation target`: total examples, first smoke batch size, and split plan.
- `Quality checks`: rules for rejecting generated examples.
- `Eval prompts`: prompts to run before and after SFT.

End with a compact synthetic-data prompt spec that another agent or script can
use to generate candidate SFT conversations. The prompt spec must include the
canonical SFT JSONL target:

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

## Defaults

Use these defaults unless the user chooses otherwise:

- Target examples: `1000`.
- First smoke batch: `100` examples.
- Split: 80% train, 10% validation, 10% test.
- Teacher model: local by default.
- Format: canonical SFT chat JSONL.
- Review: require either human review or an LLM judge before training.

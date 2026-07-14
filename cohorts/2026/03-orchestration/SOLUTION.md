# Homework 3 — AI Orchestration with Kestra

My solution for module 3. Kestra ran locally (v1.3.21) with `gemini-2.5-flash`
as the provider. The token questions were measured from the execution logs of
[`4_simple_agent`](../../../03-orchestration/flows/4_simple_agent.yaml); Q5 uses
the variant [`4_simple_agent_q5.yaml`](4_simple_agent_q5.yaml), where
`english_brevity` asks for 3 sentences instead of 1.

## Answers

| Q | Answer |
|---|--------|
| Q1 | AI Copilot has access to current Kestra plugin documentation |
| Q2 | Vague, generic, or fabricated — the model guesses from training data |
| Q3 | 60–100 output tokens |
| Q4 | 2–5x more |
| Q5 | 2–4x more |
| Q6 | Use traditional task-based workflows for predictability and auditability |

## Measured token usage

`multilingual_agent` output tokens (English, default text), across runs:

| summary_length | output tokens |
|----------------|---------------|
| short | 48 / 78 / 79 (~68) |
| long | 187 |

- **Q3** — short lands around 60–100.
- **Q4** — long (187) vs short (~68) ≈ 2.7–3.9x → 2–5x more.

`english_brevity` output tokens at `summary_length = short`:

| prompt | output tokens |
|--------|---------------|
| 1 sentence (original) | 40 / 42 / 36 (~39) |
| 3 sentences (Q5 variant) | 70 / 83 / 65 (~73) |

- **Q5** — ~73 vs ~39 ≈ 1.85x. Not "within 20%", so the closest option is 2–4x more.

## Notes

- Q1/Q2/Q6 are conceptual: the AI Copilot is grounded in current plugin docs
  (context engineering), a non-RAG model can't cite real Kestra 1.1 features, and
  regulated/deterministic workloads call for task-based flows over agents.
- Token counts vary run to run; the ranges above are from repeated executions.

## Run

```bash
cd 03-orchestration
export GEMINI_API_KEY="your-key"
export SECRET_GEMINI_API_KEY=$(echo -n "$GEMINI_API_KEY" | base64)
docker compose up -d
# import 4_simple_agent.yaml, then run it with summary_length = short / long
# import 4_simple_agent_q5.yaml and run it to compare english_brevity output
```

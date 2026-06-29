# Homework 1 — Agentic RAG

My solution for module 1. Knowledge base: the course lessons, pulled from the
repo at commit `8c1834d`. LLM: OpenAI `gpt-5.4-mini`.

Notebook: [`01-agentic-rag/code/homework_solution.ipynb`](../../../01-agentic-rag/code/homework_solution.ipynb).
Scripts: [`homework_solution.py`](../../../01-agentic-rag/code/homework_solution.py) (Q1–Q5)
and [`homework_q6.py`](../../../01-agentic-rag/code/homework_q6.py) (Q6).

## Answers

| Q | Answer |
|---|--------|
| Q1 | 72 |
| Q2 | `01-agentic-rag/lessons/14-agentic-loop.md` |
| Q3 | ~7000 (7111) |
| Q4 | 295 |
| Q5 | 3x fewer (7111 → 2294) |
| Q6 | 4 (3–4 across runs) |

## Notes

- `RAGBase` from the lessons assumes the FAQ schema, so I subclassed it: `search`
  hits our index, `build_context` joins `filename` + `content`, and `rag` returns
  the token usage so I can read `input_tokens`.
- Q5 chunks the pages (size 2000, step 1000) before indexing, which cuts the
  prompt roughly 3x.
- Q6 hands `search` to a toyaikit agent; it decides how often to call it, so the
  count moves between 3 and 4.

## Run

```bash
cd 01-agentic-rag/code
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
uv run python homework_solution.py
uv run python homework_q6.py
```

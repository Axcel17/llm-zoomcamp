# Homework 4 — Evaluation

My solution for module 4. Continues from homework 2: same 72 lesson pages
(commit `8c1834d`) and the same 295 chunks. I generate ground-truth questions
with an LLM, then compare keyword, vector and hybrid search with Hit Rate and
MRR against the provided [`ground-truth.csv`](ground-truth.csv) (360 questions).

Code lives with the module-2 project (it reuses the ONNX embedder and indexes):
[`02-vector-search/code/homework4_solution.ipynb`](../../../02-vector-search/code/homework4_solution.ipynb)
and [`homework4_solution.py`](../../../02-vector-search/code/homework4_solution.py).

## Answers

| Q | Answer |
|---|--------|
| Q1 | ~1400 input tokens |
| Q2 | `01-agentic-rag/lessons/03-rag.md` |
| Q3 | `01-agentic-rag/lessons/01-intro.md` |
| Q4 | 0.76 (hit rate) |
| Q5 | 0.55 (MRR) |
| Q6 | k = 1 |

## Measured

- **Q1** — input tokens for the first 3 pages: 1020 / 1286 / 1753 → avg ~1353.
- **Q2 / Q3** — the first ground-truth question came from `01-intro.md`. Vector
  search puts it first; keyword search returns `03-rag.md` instead. A good
  reminder that one query isn't enough — you measure across the whole set.
- **Q4** — text search: hit rate 0.758, MRR 0.594.
- **Q5** — vector search: hit rate 0.725, MRR 0.549.
- **Q6** — hybrid MRR by k: `1 → 0.648`, `50/100/200 → 0.638`. Smaller k sharpens
  the top ranks, so k = 1 wins.

## Run

```bash
cd 02-vector-search/code
uv sync
echo "OPENAI_API_KEY=sk-..." > .env
uv run python download.py   # ONNX model, if not already present
uv run python homework4_solution.py
```

# Homework 2 — Vector Search

My solution for module 2. Same 72 lesson pages as homework 1 (commit `8c1834d`),
embedded with the ONNX model `Xenova/all-MiniLM-L6-v2` through the `Embedder`
helper from the course `embed/` directory.

Notebook: [`02-vector-search/code/homework2_solution.ipynb`](../../../02-vector-search/code/homework2_solution.ipynb).
Script: [`homework2_solution.py`](../../../02-vector-search/code/homework2_solution.py).

## Answers

| Q | Answer |
|---|--------|
| Q1 | -0.02 (-0.0206) |
| Q2 | 0.37 (0.3611) |
| Q3 | `02-vector-search/lessons/07-sqlitesearch-vector.md` |
| Q4 | `04-evaluation/lessons/05-search-metrics.md` |
| Q5 | `02-vector-search/lessons/08-pgvector.md` |
| Q6 | `01-agentic-rag/lessons/13-function-calling.md` |

## Notes

- Vectors are normalized, so the dot product is the cosine similarity (Q2).
- Q3 chunks the pages, embeds every chunk into a matrix `X`, and scores the query
  with `X.dot(v)`; Q4 does the same through minsearch `VectorSearch`.
- Q5 indexes the chunks with the text `Index` and compares the top 5 of each
  method. Q6 fuses both lists with RRF (`k = 60`).

## Run

```bash
cd 02-vector-search/code
uv sync
uv run python download.py   # ONNX model into models/
uv run python homework2_solution.py
```

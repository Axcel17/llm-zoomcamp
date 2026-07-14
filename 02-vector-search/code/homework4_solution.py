"""Homework 4 (Evaluation) - Q1..Q6."""
import json

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index, VectorSearch

from embedder import Embedder
from evaluation_utils import llm_structured

load_dotenv()

GROUND_TRUTH_CSV = "../../cohorts/2026/04-evaluation/ground-truth.csv"

# ---------- Load data + chunks (same as homework 2) ----------
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]
chunks = chunk_documents(documents, size=2000, step=1000)
print("documents:", len(documents), "| chunks:", len(chunks))

embedder = Embedder()


# ---------- Q1: generating questions for the first 3 pages ----------
class Questions(BaseModel):
    questions: list[str]


data_gen_instructions = """
You emulate a student who is taking our LLM course.
You are given one lesson page from the course.
Formulate 5 questions this student might ask that are answered by this page.

Rules:
- The page should contain the answer to each question.
- Make the questions complete and not too short.
- Use as few words as possible from the page; don't copy its phrasing.
- The questions should resemble how people actually ask things online:
  not too formal, not too short, not too long.
- Ask about the content of the lesson, not about its formatting or filename.
""".strip()

client = OpenAI()
first_3 = documents[:3]
input_tokens = []
for doc in first_3:
    user_prompt = json.dumps(doc)
    _, usage = llm_structured(client, data_gen_instructions, user_prompt, Questions)
    input_tokens.append(usage.input_tokens)
print("Q1 - input tokens per call:", input_tokens)
print("Q1 - average input tokens:", sum(input_tokens) / len(input_tokens))

# ---------- Build search over the chunks ----------
text_index = Index(text_fields=["content"], keyword_fields=["filename"])
text_index.fit(chunks)

X = embedder.encode_batch([c["content"] for c in chunks])
vector_index = VectorSearch(keyword_fields=["filename"])
vector_index.fit(X, chunks)


def text_search(query, num_results=5):
    return text_index.search(query, num_results=num_results)


def vector_search(query, num_results=5):
    return vector_index.search(embedder.encode(query), num_results=num_results)


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def hybrid_search(query, k=60):
    text_results = text_search(query, num_results=10)
    vector_results = vector_search(query, num_results=10)
    return rrf([text_results, vector_results], k=k)


# ---------- Ground truth ----------
ground_truth = pd.read_csv(GROUND_TRUTH_CSV).to_dict(orient="records")
print("ground truth questions:", len(ground_truth))

q0 = ground_truth[0]["question"]
print("Q2 - text_search first result:", text_search(q0)[0]["filename"])
print("Q3 - vector_search first result:", vector_search(q0)[0]["filename"])


# ---------- Metrics (adapted to filename) ----------
def compute_relevance(q, search_function):
    correct = q["filename"]
    results = search_function(q["question"])
    return [int(d["filename"] == correct) for d in results]


def hit_rate(relevance):
    return sum(1 for line in relevance if 1 in line) / len(relevance)


def mrr(relevance):
    total = 0.0
    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total += 1 / (rank + 1)
                break
    return total / len(relevance)


def evaluate(ground_truth, search_function):
    relevance = [compute_relevance(q, search_function) for q in ground_truth]
    return {"hit_rate": hit_rate(relevance), "mrr": mrr(relevance)}


text_metrics = evaluate(ground_truth, text_search)
print("Q4 - text_search:", text_metrics)

vector_metrics = evaluate(ground_truth, vector_search)
print("Q5 - vector_search:", vector_metrics)

print("Q6 - hybrid_search MRR by k:")
best_k, best_mrr = None, -1.0
for k in [1, 50, 100, 200]:
    m = evaluate(ground_truth, lambda q, k=k: hybrid_search(q, k=k))
    print(f"    k={k}: hit_rate={m['hit_rate']:.4f} mrr={m['mrr']:.4f}")
    if m["mrr"] > best_mrr:
        best_mrr, best_k = m["mrr"], k
print("Q6 - best k:", best_k, "(mrr", round(best_mrr, 4), ")")

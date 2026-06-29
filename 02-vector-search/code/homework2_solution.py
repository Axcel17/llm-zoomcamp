"""Homework 2 (Vector Search) solution - Q1..Q6."""
import numpy as np
from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import VectorSearch, Index

from embedder import Embedder

embedder = Embedder()

# ---------- Q1: embedding a query ----------
QUERY = "How does approximate nearest neighbor search work?"
v = embedder.encode(QUERY)
print("Q1 - len(v):", len(v), "| v[0]:", round(float(v[0]), 4))

# ---------- Load the data ----------
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]
print("documents:", len(documents))

# ---------- Q2: cosine similarity ----------
page = next(
    d for d in documents
    if d["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md"
)
page_vector = embedder.encode(page["content"])
print("Q2 - cosine similarity:", round(float(page_vector.dot(v)), 4))

# ---------- Q3: chunking and search by hand ----------
chunks = chunk_documents(documents, size=2000, step=1000)
X = embedder.encode_batch([c["content"] for c in chunks])
scores = X.dot(v)
best = int(np.argmax(scores))
print("Q3 - best chunk file:", chunks[best]["filename"], "| score:", round(float(scores[best]), 4))

# ---------- Q4: vector search with minsearch ----------
vindex = VectorSearch()
vindex.fit(X, chunks)

q4_query = "What metric do we use to evaluate a search engine?"
q4_vec = embedder.encode(q4_query)
q4_results = vindex.search(q4_vec, num_results=5)
print("Q4 - first result file:", q4_results[0]["filename"])

# ---------- Q5: text search vs vector search ----------
text_index = Index(text_fields=["content"], keyword_fields=["filename"])
text_index.fit(chunks)

q5_query = "How do I store vectors in PostgreSQL?"
q5_vec = embedder.encode(q5_query)
vec_top5 = vindex.search(q5_vec, num_results=5)
txt_top5 = text_index.search(q5_query, num_results=5)

vec_files = [r["filename"] for r in vec_top5]
txt_files = [r["filename"] for r in txt_top5]
only_in_vector = [f for f in vec_files if f not in txt_files]
print("Q5 - vector top5:", vec_files)
print("Q5 - text top5  :", txt_files)
print("Q5 - in vector but not text:", only_in_vector)


# ---------- Q6: hybrid search (RRF) ----------
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


q6_query = "How do I give the model access to tools?"
q6_vec = embedder.encode(q6_query)
vector_results = vindex.search(q6_vec, num_results=5)
text_results = text_index.search(q6_query, num_results=5)
fused = rrf([vector_results, text_results])
print("Q6 - first after RRF:", fused[0]["filename"])

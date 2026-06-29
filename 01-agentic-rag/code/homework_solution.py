"""Homework 1 (Agentic RAG) - Q1..Q5."""
from dotenv import load_dotenv

load_dotenv()

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from openai import OpenAI
from rag_helper import RAGBase

QUERY = "How does the agentic loop keep calling the model until it stops?"

# ---------- Preparation ----------
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
files = reader.read()
documents = [f.parse() for f in files]

# ---------- Q1 ----------
print("Q1 - lesson pages:", len(documents))

# ---------- Q2 ----------
index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)
results = index.search(QUERY, num_results=5)
print("Q2 - first result filename:", results[0]["filename"])


# ---------- RAG adapted to our schema ----------
class RAG(RAGBase):
    def search(self, query, num_results=5):
        return self.index.search(query, num_results=num_results)

    def build_context(self, search_results):
        lines = []
        for doc in search_results:
            lines.append(doc["filename"])
            lines.append(doc["content"])
            lines.append("")
        return "\n".join(lines).strip()

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt},
        ]
        return self.llm_client.responses.create(
            model=self.model, input=input_messages
        )

    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        response = self.llm(prompt)
        return response.output_text, response.usage


client = OpenAI()

# ---------- Q3 ----------
rag = RAG(index=index, llm_client=client, model="gpt-5.4-mini")
answer, usage = rag.rag(QUERY)
print("Q3 - input tokens (full pages):", usage.input_tokens)

# ---------- Q4 ----------
chunks = chunk_documents(documents, size=2000, step=1000)
print("Q4 - number of chunks:", len(chunks))

# ---------- Q5 ----------
chunk_index = Index(text_fields=["content"], keyword_fields=["filename"])
chunk_index.fit(chunks)
rag_chunks = RAG(index=chunk_index, llm_client=client, model="gpt-5.4-mini")
answer_c, usage_c = rag_chunks.rag(QUERY)
print("Q5 - input tokens (chunked):", usage_c.input_tokens)
print("Q5 - ratio Q3/Q5:", round(usage.input_tokens / usage_c.input_tokens, 2), "x fewer")

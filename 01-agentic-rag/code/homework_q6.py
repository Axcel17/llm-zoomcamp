"""Homework 1 - Q6: turning it into an agent with toyaikit."""
from dotenv import load_dotenv

load_dotenv()

from gitsource import GithubRepositoryDataReader, chunk_documents
from minsearch import Index
from toyaikit.tools import Tools
from toyaikit.llm import OpenAIClient
from toyaikit.chat.runners import OpenAIResponsesRunner

# ---- build the chunk index (same as Q4/Q5) ----
reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [f.parse() for f in reader.read()]
chunks = chunk_documents(documents, size=2000, step=1000)

chunk_index = Index(text_fields=["content"], keyword_fields=["filename"])
chunk_index.fit(chunks)

# ---- search tool with a call counter ----
search_calls = 0


def search(query: str) -> list[dict]:
    """
    Search the course lessons for chunks matching the given query.
    """
    global search_calls
    search_calls += 1
    return chunk_index.search(query, num_results=5)


INSTRUCTIONS = (
    "You're a course teaching assistant. Answer the student's question using the "
    "search tool. Make multiple searches with different keywords before answering."
)

QUESTION = "How does the agentic loop work, and how is it different from plain RAG?"

agent_tools = Tools()
agent_tools.add_tool(search)

runner = OpenAIResponsesRunner(
    tools=agent_tools,
    developer_prompt=INSTRUCTIONS,
    llm_client=OpenAIClient(model="gpt-5.4-mini"),
)

result = runner.loop(prompt=QUESTION)

print("Q6 - search tool calls:", search_calls)

"""LLM-based restaurant recommendation using LangChain and Groq."""

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load .env from project root so GROQ_API_KEY is found regardless of cwd
_load_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_load_env_path)

PROMPT_TEMPLATE = """You are a restaurant recommendation assistant.

User request:
{query}

Restaurant options:
{context}

Choose the best restaurant for the user and explain why it fits the user's needs.

Return a concise recommendation."""


def generate_recommendation(query: str, retrieved_docs: list[Document]) -> str:
    """
    Combine retrieved restaurant documents into context, send query and context to the LLM,
    and return a concise recommendation.
    """
    if not retrieved_docs:
        return "No matching restaurants found. Try a different query or broader search."
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    chain = prompt | llm
    message = chain.invoke({"query": query, "context": context})
    return message.content if hasattr(message, "content") else str(message)

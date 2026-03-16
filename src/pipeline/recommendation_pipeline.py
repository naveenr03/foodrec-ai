"""End-to-end restaurant recommendation pipeline: retrieval + LLM."""

from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.llm.recommender import generate_recommendation


def recommend_restaurant(
    query: str,
    vector_store: FAISS,
    k: int = 5,
) -> dict[str, Any]:
    """
    Run retrieval and LLM to produce a recommendation.

    Args:
        query: Natural language search query.
        vector_store: Pre-loaded FAISS vector store.
        k: Number of restaurants to retrieve.

    Returns:
        Dict with "recommendation" (str) and "retrieved_docs" (list of Document).
    """
    retrieved_docs: list[Document] = vector_store.similarity_search(query, k=k)
    recommendation: str = generate_recommendation(query, retrieved_docs)
    return {
        "recommendation": recommendation,
        "retrieved_docs": retrieved_docs,
    }

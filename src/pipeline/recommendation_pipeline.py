"""End-to-end restaurant recommendation pipeline: retrieval + LLM."""

from __future__ import annotations

from typing import TypedDict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.llm.recommender import generate_recommendation
from src.retrieval.retriever import RetrievalHit, hits_to_documents, search_restaurants


class RecommendationResult(TypedDict):
    """Structured return from ``recommend_restaurant``."""

    recommendation: str
    retrieved_docs: list[Document]
    retrieval_hits: list[RetrievalHit]
    retrieval_scores: list[float]


def recommend_restaurant(
    query: str,
    vector_store: FAISS,
    k: int = 5,
) -> RecommendationResult:
    """
    Run retrieval and LLM to produce a recommendation.

    Args:
        query: Natural language search query.
        vector_store: Pre-loaded FAISS vector store.
        k: Number of restaurants to retrieve.

    Returns:
        Recommendation text, retrieved documents, hits with scores, and parallel score list.
    """
    hits: list[RetrievalHit] = search_restaurants(vector_store, query, k=k)
    retrieved_docs: list[Document] = hits_to_documents(hits)
    recommendation: str = generate_recommendation(query, retrieved_docs)
    return RecommendationResult(
        recommendation=recommendation,
        retrieved_docs=retrieved_docs,
        retrieval_hits=hits,
        retrieval_scores=[h.score for h in hits],
    )

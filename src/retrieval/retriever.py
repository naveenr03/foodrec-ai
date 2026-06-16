"""Semantic retriever for restaurant search using the FAISS vector store."""

from __future__ import annotations

import sys
from dataclasses import dataclass
import os
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from src.embeddings.embedder import get_embedding_model
from src.paths import default_vector_store_path
from src.retrieval.vector_store import load_vector_store

# LangChain FAISS default distance is typically Euclidean (L2): lower = closer match.
SCORE_DESCRIPTION = (
    "Vector distance from FAISS (LangChain default: L2). Lower values indicate a closer semantic match."
)


def retrieval_debug_enabled() -> bool:
    """When true, log each retrieval query, scores, and restaurant names to stderr."""
    val = os.environ.get("FOODREC_RETRIEVAL_DEBUG", "").strip().lower()
    return val in ("1", "true", "yes", "on")


def _doc_label(doc: Document) -> str:
    meta = doc.metadata or {}
    name = meta.get("name")
    if name is not None and str(name).strip():
        return str(name).strip()
    first = (doc.page_content or "").split("\n", 1)[0]
    return (first[:120] + "…") if len(first) > 120 else first or "(empty)"


@dataclass(frozen=True)
class RetrievalHit:
    """One retrieved restaurant with the similarity score returned by the vector store."""

    document: Document
    score: float


def _log_retrieval(query: str, hits: list[RetrievalHit]) -> None:
    print(f"[foodrec-retrieval] query={query!r} hits={len(hits)}", file=sys.stderr)
    for i, h in enumerate(hits, 1):
        print(
            f"[foodrec-retrieval]   {i}. score={h.score:.6g} name={_doc_label(h.document)!r}",
            file=sys.stderr,
        )


def search_restaurants(
    vector_store: FAISS,
    query: str,
    k: int = 5,
) -> list[RetrievalHit]:
    """
    Run FAISS similarity search with scores.

    Use this for all in-process retrieval so scoring and debug logging stay consistent.
    """
    pairs: list[tuple[Document, float]] = vector_store.similarity_search_with_score(
        query, k=k
    )
    hits = [RetrievalHit(document=doc, score=score) for doc, score in pairs]
    if retrieval_debug_enabled():
        _log_retrieval(query, hits)
    return hits


def retrieve_restaurants(
    query: str,
    k: int = 5,
    vector_store_path: str | Path | None = None,
) -> list[RetrievalHit]:
    """
    Load FAISS from disk and return top-k hits with scores.

    Prefer ``search_restaurants`` with a pre-loaded store in hot paths (e.g. Streamlit).
    """
    path = vector_store_path if vector_store_path is not None else default_vector_store_path()
    embedding_model = get_embedding_model()
    vector_store = load_vector_store(path, embedding_model)
    return search_restaurants(vector_store, query, k=k)


def hits_to_documents(hits: list[RetrievalHit]) -> list[Document]:
    """Strip scores for LLM context or legacy call sites."""
    return [h.document for h in hits]

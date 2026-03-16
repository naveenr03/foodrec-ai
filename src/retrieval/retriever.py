"""Semantic retriever for restaurant search using the FAISS vector store."""

from pathlib import Path

from langchain_core.documents import Document

from src.embeddings.embedder import get_embedding_model
from src.retrieval.vector_store import DEFAULT_VECTOR_STORE_PATH, load_vector_store


def retrieve_restaurants(
    query: str,
    k: int = 5,
    vector_store_path: str | Path = DEFAULT_VECTOR_STORE_PATH,
) -> list[Document]:
    """
    Load the FAISS vector store, run similarity search for the query, return top k restaurant documents.
    """
    embedding_model = get_embedding_model()
    vector_store = load_vector_store(vector_store_path, embedding_model)
    return vector_store.similarity_search(query, k=k)

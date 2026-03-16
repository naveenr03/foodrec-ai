"""FAISS vector store creation, save, and load for food-rec AI."""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

DEFAULT_VECTOR_STORE_PATH = "data/vector_store"


def create_vector_store(
    documents: list[Document],
    embedding_model: Embeddings,
) -> FAISS:
    """Create a FAISS vector store from documents and embedding model. Returns the vector store."""
    return FAISS.from_documents(documents, embedding_model)


def save_vector_store(vector_store: FAISS, path: str | Path) -> None:
    """Save the FAISS index to disk. Creates parent directories if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))


def load_vector_store(path: str | Path, embedding_model: Embeddings) -> FAISS:
    """Load a FAISS index from disk. Requires the same embedding model used when saving."""
    return FAISS.load_local(
        str(path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )

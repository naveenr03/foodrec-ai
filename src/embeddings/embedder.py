"""Embedding model setup for food-rec AI. Returns the model only; embedding is done when building the vector store."""

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Load and return the HuggingFace embedding model. Does not embed documents."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

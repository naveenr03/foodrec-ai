#!/usr/bin/env python3
"""Build and save the FAISS vector store from cleaned restaurant documents."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_dataset import load_cleaned_dataset
from src.data_processing.document_builder import build_documents
from src.embeddings.embedder import get_embedding_model
from src.retrieval.vector_store import create_vector_store, save_vector_store

VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store"


def main() -> None:
    processed_path = PROJECT_ROOT / "data" / "processed" / "restaurants_clean.csv"

    if not processed_path.exists():
        print(f"Error: Cleaned dataset not found at {processed_path}")
        print("Run scripts/run_data_pipeline.py first.")
        sys.exit(1)

    print("Loading cleaned dataset...")
    df = load_cleaned_dataset(processed_path)

    print("Building documents...")
    documents = build_documents(df=df)
    print(f"Documents: {len(documents)}")

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Creating FAISS vector store...")
    vector_store = create_vector_store(documents, embedding_model)

    print("Saving vector store...")
    save_vector_store(vector_store, VECTOR_STORE_PATH)

    print(f"Vector store created and saved to {VECTOR_STORE_PATH}")


if __name__ == "__main__":
    main()

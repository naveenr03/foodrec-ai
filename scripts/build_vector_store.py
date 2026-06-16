#!/usr/bin/env python3
"""Build and save the FAISS vector store from cleaned restaurant documents."""

import sys

import bootstrap

bootstrap.ensure_project_root()

from src.data_processing.constants import DEFAULT_RAW_CSV_NAME
from src.data_processing.document_builder import build_documents
from src.data_processing.load_dataset import load_cleaned_dataset
from src.embeddings.embedder import get_embedding_model
from src.paths import processed_restaurants_csv, project_root, vector_store_dir
from src.retrieval.vector_store import (
    create_vector_store,
    save_vector_store,
    write_index_build_manifest,
)


def main() -> None:
    processed_path = processed_restaurants_csv()
    vector_path = vector_store_dir()
    repo_root = project_root()

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
    save_vector_store(vector_store, vector_path)

    try:
        proc_rel = str(processed_path.relative_to(repo_root))
    except ValueError:
        proc_rel = str(processed_path)
    write_index_build_manifest(
        vector_path,
        processed_csv=proc_rel,
        num_documents=len(documents),
        raw_csv_name=DEFAULT_RAW_CSV_NAME,
    )
    print(f"Wrote build manifest ({DEFAULT_RAW_CSV_NAME} → {len(documents)} docs).")

    print(f"Vector store created and saved to {vector_path}")


if __name__ == "__main__":
    main()

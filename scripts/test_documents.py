#!/usr/bin/env python3
"""Test script: load dataset, generate documents, print count and one sample."""

import sys

import bootstrap

bootstrap.ensure_project_root()

from src.data_processing.document_builder import build_documents
from src.data_processing.load_dataset import load_cleaned_dataset
from src.paths import processed_restaurants_csv


def main() -> None:
    processed_path = processed_restaurants_csv()

    if not processed_path.exists():
        print(f"Error: Cleaned dataset not found at {processed_path}")
        print("Run scripts/run_data_pipeline.py first.")
        sys.exit(1)

    print("Loading cleaned dataset...")
    df = load_cleaned_dataset(processed_path)
    print(f"Loaded {len(df)} rows.")

    print("Building documents...")
    documents = build_documents(df=df)
    print(f"Created {len(documents)} documents.")

    print("\n--- Sample document ---")
    print(documents[0].page_content)
    print("\n--- End sample ---")


if __name__ == "__main__":
    main()

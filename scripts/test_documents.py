#!/usr/bin/env python3
"""Test script: load dataset, generate documents, print count and one sample."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.document_builder import build_documents, load_cleaned_dataset


def main() -> None:
    processed_path = PROJECT_ROOT / "data" / "processed" / "restaurants_clean.csv"

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

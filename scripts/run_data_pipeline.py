#!/usr/bin/env python3
"""Runnable script to execute the data processing pipeline and generate the cleaned dataset."""

import sys
from pathlib import Path

# Add project root so we can import from src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_dataset import run_pipeline


def main() -> None:
    raw_path = PROJECT_ROOT / "data" / "raw" / "zomato.csv"
    processed_path = PROJECT_ROOT / "data" / "processed" / "restaurants_clean.csv"

    if not raw_path.exists():
        print(f"Error: Raw dataset not found at {raw_path}")
        sys.exit(1)

    print("Running data processing pipeline...")
    df = run_pipeline(
        raw_path=raw_path,
        processed_path=processed_path,
        sample_size=2000,
        random_state=42,
    )
    print(f"Done. Cleaned dataset: {len(df)} rows saved to {processed_path}")


if __name__ == "__main__":
    main()

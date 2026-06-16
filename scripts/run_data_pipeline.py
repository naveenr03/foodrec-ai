#!/usr/bin/env python3
"""Runnable script to execute the data processing pipeline and generate the cleaned dataset.

Uses ``DEFAULT_RAW_CSV_NAME`` from ``src.data_processing.constants`` (via ``src.paths.raw_dataset_csv``).
"""

import sys

import bootstrap

bootstrap.ensure_project_root()

from src.data_processing.load_dataset import run_pipeline
from src.paths import processed_restaurants_csv, raw_dataset_csv


def main() -> None:
    raw_path = raw_dataset_csv()
    processed_path = processed_restaurants_csv()

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
    print(
        f"Done. Cleaned dataset: {len(df)} rows, {len(df.columns)} columns saved to {processed_path}"
    )
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()

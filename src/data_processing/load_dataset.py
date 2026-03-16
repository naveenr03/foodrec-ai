"""Load and clean the Zomato restaurant dataset for the food-rec AI pipeline."""

from pathlib import Path

import pandas as pd

# Column names as in the raw CSV
COLUMNS = [
    "name",
    "location",
    "cuisines",
    "approx_cost(for two people)",
    "rate",
    "rest_type",
]
COST_COLUMN = "approx_cost(for two people)"


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the restaurant dataset from CSV."""
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset: keep only required columns, drop nulls, convert cost to numeric.
    """
    # Keep only required columns (ignore missing columns for flexibility)
    available = [c for c in COLUMNS if c in df.columns]
    df = df[available].copy()

    # Convert cost to numeric (handle commas and non-numeric values)
    if COST_COLUMN in df.columns:
        cost_series = df[COST_COLUMN].astype(str).str.replace(",", "", regex=False)
        df[COST_COLUMN] = pd.to_numeric(cost_series, errors="coerce")

    # Remove rows with null values
    df = df.dropna()

    return df


def sample_rows(df: pd.DataFrame, n: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Sample n rows for experimentation. Uses fixed random_state for reproducibility."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state)


def save_cleaned_dataset(df: pd.DataFrame, path: str | Path) -> None:
    """Save the cleaned dataset to CSV. Creates parent directories if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_pipeline(
    raw_path: str | Path = "data/raw/zomato.csv",
    processed_path: str | Path = "data/processed/restaurants_clean.csv",
    sample_size: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run the full dataset processing pipeline: load, clean, sample, save.
    Returns the cleaned and sampled DataFrame.
    """
    df = load_raw_dataset(raw_path)
    df = clean_dataset(df)
    df = sample_rows(df, n=sample_size, random_state=random_state)
    save_cleaned_dataset(df, processed_path)
    return df

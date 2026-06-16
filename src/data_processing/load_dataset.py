"""Load and clean the Zomato restaurant dataset for the food-rec AI pipeline."""

from pathlib import Path

import pandas as pd

from src.data_processing.column_aliases import (
    COLUMNS,
    COST_COLUMN,
    VARIANT_TO_CANONICAL,
    normalize_header_key,
)
from src.paths import processed_restaurants_csv, raw_dataset_csv

# Re-export for callers that imported schema from this module
__all__ = [
    "COLUMNS",
    "COST_COLUMN",
    "load_raw_dataset",
    "load_cleaned_dataset",
    "clean_dataset",
    "sample_rows",
    "save_cleaned_dataset",
    "run_pipeline",
]


def _coerce_numeric_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """In-place numeric coercion; cost column strips commas first."""
    for col in columns:
        if col not in df.columns:
            continue
        if col == COST_COLUMN:
            series = df[col].astype(str).str.replace(",", "", regex=False)
            df[col] = pd.to_numeric(series, errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the restaurant dataset from CSV."""
    return pd.read_csv(path)


def _strip_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lstrip("\ufeff\u200b") for c in out.columns]
    return out


def _apply_case_fold_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Map Title Case / UPPER headers (e.g. ``Name``) to the lowercase names used in code."""
    want_lower = {c.lower(): c for c in COLUMNS}
    renames: dict[str, str] = {}
    for col in df.columns:
        lw = col.lower()
        if lw not in want_lower:
            continue
        target = want_lower[lw]
        if col == target:
            continue
        if target in df.columns:
            continue
        renames[col] = target
    return df.rename(columns=renames)


def _apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename known alternate headers to canonical Zomato-style names.

    Headers are matched with light normalization (see ``column_aliases.normalize_header_key``).
    Longer patterns are tried first so ``restaurant name`` wins over ``restaurant``.
    """
    df = df.copy()
    pairs = [
        (normalize_header_key(k), v, len(k))
        for k, v in VARIANT_TO_CANONICAL.items()
    ]
    pairs.sort(key=lambda t: t[2], reverse=True)
    for pat_norm, target, _ in pairs:
        for col in list(df.columns):
            if normalize_header_key(col) != pat_norm:
                continue
            if col == target:
                break
            if target in df.columns:
                break
            df = df.rename(columns={col: target})
            break
    return df


def load_cleaned_dataset(path: str | Path = "data/processed/restaurants_clean.csv") -> pd.DataFrame:
    """Load the cleaned restaurant dataset from CSV."""
    path = Path(path)
    if path.is_file() and path.stat().st_size == 0:
        raise ValueError(
            f"{path} is empty. Delete it and re-run `python scripts/run_data_pipeline.py` "
            "(the last run may have produced an invalid file if the raw CSV had no matching columns)."
        )
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError as e:
        raise ValueError(
            f"{path} has no readable columns (often caused by a raw CSV whose headers do not match "
            f"the expected schema). Expected columns include: {COLUMNS}. "
            "Re-run `python scripts/run_data_pipeline.py` after fixing the raw file or extending "
            "aliases in `src/data_processing/column_aliases.py`."
        ) from e


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset: keep only required columns, drop nulls, convert cost to numeric.
    """
    df = _strip_column_names(df)
    df = _apply_case_fold_to_canonical(df)
    df = _apply_column_aliases(df)

    available = [c for c in COLUMNS if c in df.columns]
    if not available:
        raise ValueError(
            "Raw CSV has no recognized required columns after alias mapping. "
            f"Expected at least one of: {COLUMNS}. "
            f"Columns in file (after strip): {list(df.columns)!r}"
        )
    if "name" not in available:
        raise ValueError(
            "Could not map any column to restaurant **name** (required for search and documents). "
            f"Recognized columns so far: {available!r}. "
            f"All columns in the raw file: {list(df.columns)!r}. "
            "Rename your CSV header to `name` or add a matching entry in "
            "`VARIANT_TO_CANONICAL` in `src/data_processing/column_aliases.py`."
        )
    if len(available) < 3:
        raise ValueError(
            f"Too few usable columns after mapping ({len(available)}): {available!r}. "
            "At least name, location, and one of cuisines/cost/rate/rest_type are expected for useful retrieval. "
            f"Raw columns were: {list(df.columns)!r}."
        )
    df = df[available].copy()

    numeric_cols = [c for c in (COST_COLUMN, "rate") if c in df.columns]
    _coerce_numeric_columns(df, numeric_cols)

    df = df.dropna()

    if df.empty:
        raise ValueError(
            "After cleaning, no rows remain (null values remain in one or more of: "
            f"{available!r}). Check the raw CSV or relax null handling."
        )

    for c in COLUMNS:
        if c not in df.columns:
            df[c] = pd.NA
    df = df[COLUMNS]

    _coerce_numeric_columns(df, [c for c in (COST_COLUMN, "rate") if c in df.columns])

    return df


def sample_rows(df: pd.DataFrame, n: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """Sample n rows for experimentation. Uses fixed random_state for reproducibility."""
    if len(df) <= n:
        return df
    return df.sample(n=n, random_state=random_state)


def save_cleaned_dataset(df: pd.DataFrame, path: str | Path) -> None:
    """Save the cleaned dataset to CSV. Creates parent directories if needed."""
    if df.shape[1] == 0:
        raise ValueError(
            "Refusing to save cleaned CSV: zero columns (raw file headers likely do not match "
            f"expected columns {COLUMNS})."
        )
    if df.shape[0] == 0:
        raise ValueError("Refusing to save cleaned CSV: zero rows after cleaning.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_pipeline(
    raw_path: str | Path | None = None,
    processed_path: str | Path | None = None,
    sample_size: int = 2000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Run the full dataset processing pipeline: load, clean, sample, save.
    Returns the cleaned and sampled DataFrame.
    """
    raw = raw_path if raw_path is not None else raw_dataset_csv()
    processed = processed_path if processed_path is not None else processed_restaurants_csv()
    df = load_raw_dataset(raw)
    df = clean_dataset(df)
    df = sample_rows(df, n=sample_size, random_state=random_state)
    save_cleaned_dataset(df, processed)
    return df

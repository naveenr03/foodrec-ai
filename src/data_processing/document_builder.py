"""Build LangChain Document objects from the cleaned restaurant dataset."""

from pathlib import Path

import pandas as pd
from langchain_core.documents import Document


def load_cleaned_dataset(path: str | Path = "data/processed/restaurants_clean.csv") -> pd.DataFrame:
    """Load the cleaned restaurant dataset from CSV."""
    return pd.read_csv(path)


def _row_to_page_content(row: pd.Series) -> str:
    """Format a single restaurant row as the document text."""
    name = row.get("name", "")
    location = row.get("location", "")
    cuisines = row.get("cuisines", "")
    cost = row.get("approx_cost(for two people)", "")
    rating = row.get("rate", "")
    rest_type = row.get("rest_type", "")

    return (
        f"Restaurant: {name}\n"
        f"Location: {location}\n"
        f"Cuisine: {cuisines}\n"
        f"Cost for two: {cost} INR\n"
        f"Rating: {rating}\n"
        f"Type: {rest_type}"
    )


def build_documents(df: pd.DataFrame | None = None, path: str | Path | None = None) -> list[Document]:
    """
    Convert restaurant rows into LangChain Document objects.

    Either pass a DataFrame (df) or a path to the cleaned CSV (path).
    If both are None, loads from data/processed/restaurants_clean.csv.
    """
    if df is None:
        load_path = path if path is not None else "data/processed/restaurants_clean.csv"
        df = load_cleaned_dataset(load_path)

    documents: list[Document] = []
    for _, row in df.iterrows():
        page_content = _row_to_page_content(row)
        metadata = {
            "name": row.get("name"),
            "location": row.get("location"),
            "cuisines": row.get("cuisines"),
            "approx_cost": row.get("approx_cost(for two people)"),
            "rate": row.get("rate"),
            "rest_type": row.get("rest_type"),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents

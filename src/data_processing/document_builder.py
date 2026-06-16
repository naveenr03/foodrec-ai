"""Build LangChain Document objects from the cleaned restaurant dataset."""

import ast
import math
from pathlib import Path

import pandas as pd
from langchain_core.documents import Document

from src.data_processing.load_dataset import load_cleaned_dataset


def _is_missing(val: object) -> bool:
    if val is None:
        return True
    try:
        if pd.isna(val):
            return True
    except (TypeError, ValueError):
        pass
    s = str(val).strip().lower()
    return s in ("", "nan", "none", "<na>", "nat")


def _format_scalar(val: object) -> str | None:
    """Human-readable scalar for document text; None if missing."""
    if _is_missing(val):
        return None
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        if isinstance(val, float) and val == int(val):
            return str(int(val))
        return str(val)
    return str(val).strip() or None


def _format_list_like(val: object) -> str | None:
    """Turn list-like strings (e.g. Zomato ``['Biryani', ' North Indian']``) into comma-separated text."""
    if _is_missing(val):
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        if s.startswith("["):
            parsed = ast.literal_eval(s)
            if isinstance(parsed, (list, tuple)):
                parts = [str(x).strip() for x in parsed if str(x).strip()]
                return ", ".join(parts) if parts else None
    except (ValueError, SyntaxError, TypeError):
        pass
    return s.strip("[]'\"") or None


def _row_to_page_content(row: pd.Series) -> str:
    """Format a single restaurant row as the document text (embedding + LLM context)."""
    name = _format_scalar(row.get("name")) or "Unknown"
    location = _format_scalar(row.get("location"))
    cuisines = _format_list_like(row.get("cuisines")) or _format_scalar(row.get("cuisines"))
    cost = _format_scalar(row.get("approx_cost(for two people)"))
    rating = _format_scalar(row.get("rate"))
    rest_type = _format_list_like(row.get("rest_type")) or _format_scalar(row.get("rest_type"))

    lines: list[str] = [f"Restaurant: {name}"]
    if location:
        lines.append(f"Location: {location}")
    if cuisines:
        lines.append(f"Cuisine: {cuisines}")
    if cost:
        lines.append(f"Cost for two: {cost} INR")
    if rating:
        lines.append(f"Rating: {rating}")
    if rest_type:
        lines.append(f"Type / features: {rest_type}")
    return "\n".join(lines)


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

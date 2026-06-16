"""Resolved paths from the repository root (parent of ``src``)."""

from __future__ import annotations

from pathlib import Path

from src.data_processing.constants import DEFAULT_RAW_CSV_NAME

# String default when callers run with cwd = repo root (legacy LangChain examples)
VECTOR_STORE_RELATIVE = "data/vector_store"


def project_root() -> Path:
    """Repository root (directory that contains ``src`` and ``data``)."""
    return Path(__file__).resolve().parent.parent


def processed_restaurants_csv() -> Path:
    return project_root() / "data" / "processed" / "restaurants_clean.csv"


def vector_store_dir() -> Path:
    return project_root() / "data" / "vector_store"


def raw_dataset_csv(filename: str | None = None) -> Path:
    """Path to the raw restaurant CSV under ``data/raw/``."""
    name = filename if filename is not None else DEFAULT_RAW_CSV_NAME
    return project_root() / "data" / "raw" / name


def default_vector_store_path() -> Path:
    """Absolute path to the FAISS directory (same location as ``VECTOR_STORE_RELATIVE`` from root)."""
    return vector_store_dir()

"""FAISS vector store creation, save, and load for food-rec AI."""

import json
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.paths import VECTOR_STORE_RELATIVE

# Relative path when process cwd is the repo root (legacy callers / docs)
DEFAULT_VECTOR_STORE_PATH = VECTOR_STORE_RELATIVE
INDEX_BUILD_MANIFEST = "index_build_manifest.json"


def vector_store_fingerprint(path: str | Path) -> int:
    """
    Latest file mtime (nanoseconds) under the vector store dir, excluding the manifest.

    Used to bust Streamlit's ``@st.cache_resource`` when the index is rebuilt on disk.
    """
    path = Path(path)
    if not path.is_dir():
        return 0
    best = 0
    for child in path.rglob("*"):
        if not child.is_file() or child.name == INDEX_BUILD_MANIFEST:
            continue
        try:
            best = max(best, child.stat().st_mtime_ns)
        except OSError:
            continue
    return best


def write_index_build_manifest(
    vector_store_dir: str | Path,
    *,
    processed_csv: str,
    num_documents: int,
    raw_csv_name: str,
) -> None:
    """Record which processed CSV and raw filename were used when building the FAISS index."""
    from datetime import datetime, timezone

    vector_store_dir = Path(vector_store_dir)
    payload = {
        "source_processed_csv": processed_csv,
        "processed_row_count": num_documents,
        "raw_csv_name_at_build": raw_csv_name,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    out = vector_store_dir / INDEX_BUILD_MANIFEST
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def read_index_build_manifest(vector_store_dir: str | Path) -> dict | None:
    """Return manifest dict if present, else None."""
    path = Path(vector_store_dir) / INDEX_BUILD_MANIFEST
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def create_vector_store(
    documents: list[Document],
    embedding_model: Embeddings,
) -> FAISS:
    """Create a FAISS vector store from documents and embedding model. Returns the vector store."""
    return FAISS.from_documents(documents, embedding_model)


def save_vector_store(vector_store: FAISS, path: str | Path) -> None:
    """Save the FAISS index to disk. Creates parent directories if needed."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))


def load_vector_store(path: str | Path, embedding_model: Embeddings) -> FAISS:
    """Load a FAISS index from disk. Requires the same embedding model used when saving."""
    return FAISS.load_local(
        str(path),
        embedding_model,
        allow_dangerous_deserialization=True,
    )

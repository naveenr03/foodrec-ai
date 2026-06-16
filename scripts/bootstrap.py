"""Ensure repository root is on ``sys.path`` so ``src.*`` imports work from CLI scripts."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root() -> Path:
    """Parent of ``scripts/`` (repo root). Inserts it at the front of ``sys.path`` if missing."""
    root = Path(__file__).resolve().parent.parent
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root

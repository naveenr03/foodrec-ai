"""Build the natural-language query string from structured preference inputs (no Streamlit)."""


def build_search_query(
    food_dish: str,
    cuisine: str,
    budget: float | int,
    ambience: str,
    group_size: int,
) -> str:
    """
    Combine optional fields into one comma-separated query for the retriever and LLM.

    Empty or whitespace-only strings are ignored; ``budget`` 0 is ignored.
    """
    parts: list[str] = []
    fd = (food_dish or "").strip()
    if fd:
        parts.append(f"looking for {fd}")
    if (cuisine or "").strip():
        parts.append(f"{cuisine.strip()} cuisine")
    if budget and float(budget) > 0:
        parts.append(f"budget around {int(budget)} INR for two")
    if (ambience or "").strip():
        parts.append(f"{ambience.strip()} ambience")
    if group_size and int(group_size) > 0:
        parts.append(f"for group of {group_size}")
    return ", ".join(parts) if parts else "good restaurant"

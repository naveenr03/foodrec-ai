"""Streamlit presentation helpers (styles, hero, results)."""

from __future__ import annotations

import streamlit as st

from src.pipeline.recommendation_pipeline import RecommendationResult
from src.retrieval.retriever import SCORE_DESCRIPTION


def render_global_styles() -> None:
    """Inject scoped CSS for layout and hero strip."""
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.5rem; max-width: 880px; }
        .hero-strip {
            background: linear-gradient(120deg, #0f766e 0%, #0d9488 45%, #14b8a6 100%);
            color: white;
            padding: 1.5rem 1.35rem;
            border-radius: 12px;
            margin-bottom: 1.25rem;
            box-shadow: 0 4px 14px rgba(15, 118, 110, 0.25);
        }
        .hero-strip h1 { margin: 0 0 0.4rem 0; font-size: 1.75rem; font-weight: 700;
            letter-spacing: -0.02em; color: white !important; border: none !important; }
        .hero-strip p { margin: 0; opacity: 0.95; font-size: 1rem; line-height: 1.55;
            color: rgba(255,255,255,0.95) !important; }
        [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input { border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-strip">
            <h1>🍽️ Foodrec-ai</h1>
            <p>Restaurant picks from semantic search + AI. Name a dish (e.g. burger) or cuisine—then refine with budget and ambience.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_index_manifest_caption(manifest: dict | None) -> None:
    if manifest:
        raw_b = manifest.get("raw_csv_name_at_build", "?")
        n = manifest.get("processed_row_count", "?")
        when = manifest.get("built_at_utc", "")
        st.caption(
            f"**Search index:** built from `{raw_b}` → **{n}** restaurants in the index "
            f"(processed CSV: `{manifest.get('source_processed_csv', '?')}`). "
            f"Built (UTC): {when}. "
            "If this is the wrong city, run `python scripts/run_data_pipeline.py` then "
            "`python scripts/build_vector_store.py`, then restart Streamlit."
        )
    else:
        st.caption(
            "No `index_build_manifest.json` in the vector store folder — run "
            "`python scripts/build_vector_store.py` after the data pipeline so the UI can show "
            "which dataset the index came from."
        )


def render_preferences_card() -> tuple[str, str, float, str, int, bool]:
    """
    Render the bordered preference form and primary button.

    Returns:
        food_dish, cuisine, budget, ambience, group_size, find_btn_clicked
    """
    with st.container(border=True):
        st.markdown("**Your preferences**")
        food_dish = st.text_input(
            "Food or dish (optional)",
            placeholder="e.g. burger, biryani, pizza",
            help="Search for a specific food or dish. The retriever uses this to find matching restaurants.",
            key="food_dish",
        )
        col1, col2 = st.columns(2, gap="large")
        with col1:
            cuisine = st.text_input(
                "Cuisine (optional)",
                placeholder="e.g. North Indian, Chinese",
                help="Broad cuisine style or region—leave blank if you only care about the dish above.",
                key="cuisine",
            )
            budget = st.number_input(
                "Budget (₹ for two)",
                min_value=0,
                value=500,
                step=100,
                help="Approximate spend for two people.",
                key="budget",
            )
        with col2:
            ambience = st.text_input(
                "Ambience",
                placeholder="e.g. casual, quiet, rooftop",
                help="Vibe or setting you prefer.",
                key="ambience",
            )
            group_size = st.number_input(
                "Group size",
                min_value=1,
                value=2,
                step=1,
                help="How many people are dining?",
                key="group_size",
            )

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        find_btn = st.button("Find restaurant", type="primary", use_container_width=True)

    return food_dish, cuisine, budget, ambience, group_size, find_btn


def render_search_results(result: RecommendationResult, query_used: str) -> None:
    """Show recommendation, explanation, and retrieved restaurant expanders."""
    recommendation = result["recommendation"]
    retrieved_docs = result["retrieved_docs"]

    st.divider()
    st.caption(f"**Query used:** {query_used}")

    st.markdown("### Recommended restaurant")
    st.success(recommendation)

    st.markdown("### Explanation")
    st.info(
        "The suggestion is based on your inputs and the closest matches in our dataset. "
        "Expand the cards below for full restaurant details."
    )

    st.markdown("### Retrieved restaurants")
    st.caption(f"{len(retrieved_docs)} matches from the vector index. {SCORE_DESCRIPTION}")
    retrieval_hits = result.get("retrieval_hits")
    if retrieval_hits:
        for i, hit in enumerate(retrieval_hits, 1):
            with st.expander(
                f"**Option {i}** — score `{hit.score:.6g}` (lower = closer match)",
                expanded=(i == 1),
            ):
                st.markdown(hit.document.page_content.replace("\n", "\n\n"))
    else:
        for i, doc in enumerate(retrieved_docs, 1):
            with st.expander(f"**Option {i}** — details", expanded=(i == 1)):
                st.markdown(doc.page_content.replace("\n", "\n\n"))

#!/usr/bin/env python3
"""Foodrec-ai - Streamlit UI for restaurant recommendations."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.embeddings.embedder import get_embedding_model
from src.pipeline.recommendation_pipeline import recommend_restaurant
from src.retrieval.vector_store import load_vector_store

VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store"

st.set_page_config(
    page_title="Foodrec-ai - Restaurant Recommendation",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Scoped polish: readable width, hero strip, rounded inputs (works with light & dark Streamlit themes)
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
    .hero-strip h1 { margin: 0 0 0.4rem 0; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; color: white !important; border: none !important; }
    .hero-strip p { margin: 0; opacity: 0.95; font-size: 1rem; line-height: 1.55; color: rgba(255,255,255,0.95) !important; }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-strip">
        <h1>🍽️ Foodrec-ai</h1>
        <p>Restaurant picks from semantic search + AI. Name a dish (e.g. burger) or cuisine—then refine with budget and ambience.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load embedding model and FAISS vector store once at app start (cached for the session)
@st.cache_resource
def load_embedding_model():
    return get_embedding_model()


@st.cache_resource
def load_vector_database(embedding_model):
    return load_vector_store(VECTOR_STORE_PATH, embedding_model)


try:
    embedding_model = load_embedding_model()
    vector_store = load_vector_database(embedding_model)
except Exception as e:
    st.error(
        f"Could not load vector store. Run `scripts/build_vector_store.py` first and ensure "
        f"`data/vector_store` exists. Error: {e}"
    )
    st.stop()

# --- Preferences ---
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

if find_btn:
    parts = []
    fd = (food_dish or "").strip()
    if fd:
        parts.append(f"looking for {fd}")
    if cuisine:
        parts.append(f"{cuisine} cuisine")
    if budget and budget > 0:
        parts.append(f"budget around {int(budget)} INR for two")
    if ambience:
        parts.append(f"{ambience} ambience")
    if group_size:
        parts.append(f"for group of {group_size}")
    query = ", ".join(parts) if parts else "good restaurant"

    with st.spinner("Searching the index and drafting your recommendation…"):
        result = recommend_restaurant(query, vector_store, k=5)

    st.session_state["reco_result"] = result
    st.session_state["reco_query"] = query

# --- Results (persist until next search) ---
if st.session_state.get("reco_result"):
    result = st.session_state["reco_result"]
    query_used = st.session_state.get("reco_query", "")
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
    st.caption(f"{len(retrieved_docs)} matches from the vector index")
    for i, doc in enumerate(retrieved_docs, 1):
        with st.expander(f"**Option {i}** — details", expanded=(i == 1)):
            st.markdown(doc.page_content.replace("\n", "\n\n"))

#!/usr/bin/env python3
"""Foodrec-ai - Streamlit UI for restaurant recommendations."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.query_builder import build_search_query
from app.ui import (
    render_global_styles,
    render_hero,
    render_index_manifest_caption,
    render_preferences_card,
    render_search_results,
)
from src.embeddings.embedder import get_embedding_model
from src.paths import vector_store_dir
from src.pipeline.recommendation_pipeline import recommend_restaurant
from src.retrieval.vector_store import (
    load_vector_store,
    read_index_build_manifest,
    vector_store_fingerprint,
)

VECTOR_STORE_PATH = vector_store_dir()

st.set_page_config(
    page_title="Foodrec-ai - Restaurant Recommendation",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_global_styles()
render_hero()


@st.cache_resource
def load_embedding_model():
    return get_embedding_model()


@st.cache_resource
def load_vector_database(embedding_model, index_fingerprint: int):
    return load_vector_store(VECTOR_STORE_PATH, embedding_model)


_index_fp = vector_store_fingerprint(VECTOR_STORE_PATH)

try:
    embedding_model = load_embedding_model()
    vector_store = load_vector_database(embedding_model, _index_fp)
except Exception as e:
    st.error(
        f"Could not load vector store. Run `scripts/build_vector_store.py` first and ensure "
        f"`data/vector_store` exists. Error: {e}"
    )
    st.stop()

render_index_manifest_caption(read_index_build_manifest(VECTOR_STORE_PATH))

food_dish, cuisine, budget, ambience, group_size, find_btn = render_preferences_card()

if find_btn:
    query = build_search_query(food_dish, cuisine, budget, ambience, group_size)
    with st.spinner("Searching the index and drafting your recommendation…"):
        result = recommend_restaurant(query, vector_store, k=5)
    st.session_state["reco_result"] = result
    st.session_state["reco_query"] = query

if st.session_state.get("reco_result"):
    render_search_results(
        st.session_state["reco_result"],
        st.session_state.get("reco_query", ""),
    )

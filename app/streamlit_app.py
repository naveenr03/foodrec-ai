#!/usr/bin/env python3
"""GroupBite AI - Streamlit UI for restaurant recommendations."""

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
    page_title="GroupBite AI - Restaurant Recommendation",
    page_icon="🍽️",
    layout="centered",
)

st.title("GroupBite AI - Restaurant Recommendation")
st.markdown("Tell us your preferences and we'll suggest a restaurant.")

# Load embedding model and FAISS vector store once at app start (cached for the session)
@st.cache_resource
def load_embedding_model():
    return get_embedding_model()


@st.cache_resource
def load_vector_database(embedding_model):
    return load_vector_store(VECTOR_STORE_PATH, embedding_model)


# Load resources on first run
try:
    embedding_model = load_embedding_model()
    vector_store = load_vector_database(embedding_model)
except Exception as e:
    st.error(
        f"Could not load vector store. Run `scripts/build_vector_store.py` first and ensure "
        f"`data/vector_store` exists. Error: {e}"
    )
    st.stop()

# Inputs
col1, col2 = st.columns(2)
with col1:
    cuisine = st.text_input("Cuisine preference", placeholder="e.g. North Indian, Biryani")
    budget = st.number_input("Budget (INR for two)", min_value=0, value=500, step=100)
with col2:
    ambience = st.text_input("Ambience", placeholder="e.g. casual, quiet, rooftop")
    group_size = st.number_input("Group size", min_value=1, value=2, step=1)

find_btn = st.button("Find Restaurant")

if find_btn:
    # Build query from inputs
    parts = []
    if cuisine:
        parts.append(f"{cuisine} cuisine")
    if budget and budget > 0:
        parts.append(f"budget around {int(budget)} INR for two")
    if ambience:
        parts.append(f"{ambience} ambience")
    if group_size:
        parts.append(f"for group of {group_size}")
    query = ", ".join(parts) if parts else "good restaurant"

    with st.spinner("Finding a restaurant for you..."):
        result = recommend_restaurant(query, vector_store, k=5)

    recommendation = result["recommendation"]
    retrieved_docs = result["retrieved_docs"]

    st.divider()

    # Recommended Restaurant
    st.subheader("Recommended Restaurant")
    st.success(recommendation)

    # Explanation
    st.subheader("Explanation")
    st.info(
        "The recommendation above explains why this restaurant fits your preferences, "
        "budget, and group size."
    )

    # Retrieved Restaurants
    st.subheader("Retrieved Restaurants")
    for i, doc in enumerate(retrieved_docs, 1):
        with st.expander(f"Option {i}"):
            st.text(doc.page_content)

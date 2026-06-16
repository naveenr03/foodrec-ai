"""Canonical restaurant CSV schema and header alias map for raw Zomato-style exports.

Edit this file when a new dataset uses different column names; keep ETL logic in ``load_dataset``.
"""

from __future__ import annotations

# Output / in-app column names (order preserved when padding processed CSV)
COLUMNS: list[str] = [
    "name",
    "location",
    "cuisines",
    "approx_cost(for two people)",
    "rate",
    "rest_type",
]

COST_COLUMN = "approx_cost(for two people)"

# Normalized keys (see ``normalize_header_key``) -> canonical name in ``COLUMNS``
VARIANT_TO_CANONICAL: dict[str, str] = {
    # --- name ---
    "restaurant_name": "name",
    "restaurant name": "name",
    "restaurant": "name",
    "res_name": "name",
    "res name": "name",
    "title": "name",
    "establishment": "name",
    "establishment_name": "name",
    "establishment name": "name",
    "shop_name": "name",
    "shop name": "name",
    "name_of_restaurant": "name",
    "name of restaurant": "name",
    "restaurantname": "name",
    "venue": "name",
    "hotel_name": "name",
    "hotel name": "name",
    # --- location (only used if canonical "location" is absent) ---
    "locality": "location",
    "area": "location",
    "neighborhood": "location",
    "address": "location",
    "city": "location",
    "location_name": "location",
    "location name": "location",
    "place": "location",
    "region": "location",
    # --- cuisines ---
    "cuisine": "cuisines",
    "food_type": "cuisines",
    "food type": "cuisines",
    "food": "cuisines",
    "type_of_food": "cuisines",
    "type of food": "cuisines",
    # --- cost ---
    "approx cost for two people": "approx_cost(for two people)",
    "approx cost(for two people)": "approx_cost(for two people)",
    "approx_cost(for two people)": "approx_cost(for two people)",
    "cost for two": "approx_cost(for two people)",
    "approx cost": "approx_cost(for two people)",
    "cost": "approx_cost(for two people)",
    "price": "approx_cost(for two people)",
    "cost_for_two": "approx_cost(for two people)",
    "cost for two people": "approx_cost(for two people)",
    "average_cost": "approx_cost(for two people)",
    "avg_cost": "approx_cost(for two people)",
    "price for 2": "approx_cost(for two people)",
    "price for two": "approx_cost(for two people)",
    "price for 2 people": "approx_cost(for two people)",
    "cost for 2": "approx_cost(for two people)",
    # --- rate ---
    "dining rating": "rate",
    "aggregate_rating": "rate",
    "aggregate rating": "rate",
    "rating": "rate",
    "ratings": "rate",
    "stars": "rate",
    "score": "rate",
    "avg_rating": "rate",
    "mean_rating": "rate",
    "google_rating": "rate",
    # --- rest_type ---
    "listed_in(type)": "rest_type",
    "listed in(type)": "rest_type",
    "category": "rest_type",
    "restaurant_type": "rest_type",
    "restaurant type": "rest_type",
    "establishment_type": "rest_type",
    "establishment type": "rest_type",
    "features": "rest_type",
    "amenities": "rest_type",
}


def normalize_header_key(s: str) -> str:
    """Lowercase, strip, collapse punctuation so ``Restaurant-Name`` matches ``restaurant name``."""
    t = str(s).strip().lower()
    for ch in "-_()./":
        t = t.replace(ch, " ")
    return " ".join(t.split())

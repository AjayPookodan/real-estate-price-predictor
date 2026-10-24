# src/features.py

import numpy as np
import pandas as pd

# Add any other imports required by your existing functions.
ACC_COLS = [
    "dist_railway_km",
    "dist_bus_km",
    "dist_school_km",
    "dist_hospital_km",
    "dist_mall_km",
    "dist_highway_km"
]

WEIGHTS = {
    "railway": 0.20,
    "hospital": 0.20,
    "highway": 0.15,
    "school": 0.15,
    "mall": 0.15,
    "bus": 0.15
}

COUNT_BASES = ["school", "hospital", "bus", "mall"]

# --------------------------------------------------
# 1. Basic features
# --------------------------------------------------

def add_basic_features(d):
    d = d.copy()

    # 1. Bathroom-to-bedroom ratio
    d["bed_bath_ratio"] = (
        d["bathroom_num"] / d["bedroom_num"]
    )

    # 2. Total rooms
    d["total_rooms"] = (
        d["bedroom_num"]
        + d["bathroom_num"]
        + d["balcony_num"]
    )

    # 3. Area available per bedroom
    d["area_per_bedroom"] = (
        d["area_sqft"] / d["bedroom_num"]
    )

    # 4. Floor position relative to the building
    d["floor_ratio"] = (
        d["floor_num"]
        / d["total_floor"].replace(0, np.nan)
    ).clip(0, 1)

    return d


# --------------------------------------------------
# 2. Accessibility features
# --------------------------------------------------

def fit_access_params(train, cols):
    p = {}

    for c in cols:
        x = np.log1p(train[c])

        lo = float(x.quantile(0.05))
        hi = float(x.quantile(0.95))

        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(
                f"Invalid normalization bounds for {c}: lo={lo}, hi={hi}"
            )

        if hi <= lo:
            raise ValueError(
                f"Cannot normalize {c}: upper bound must exceed lower bound. "
                f"lo={lo}, hi={hi}"
            )

        p[c] = {
            "lo": lo,
            "hi": hi
        }

    return p

def access_components(d, params):
    out = pd.DataFrame(index=d.index)

    for c, q in params.items():
        x = np.log1p(d[c])

        normalized = (
            (x - q["lo"]) / (q["hi"] - q["lo"])
        ).clip(0, 1)

        out[f"acc_{c[5:-3]}"] = 1 - normalized

    return out

def add_access_features(d, params, weights):
    comp = access_components(d, params)

    d = d.copy()

    # Equal-weight accessibility score
    d["access_plain"] = comp.mean(axis=1)

    # Keep only weights for components that actually exist
    available_weights = {
        f"acc_{k}": v
        for k, v in weights.items()
        if f"acc_{k}" in comp.columns
    }

    if not available_weights:
        raise ValueError(
            "No accessibility components match the supplied weights."
        )

    w = pd.Series(available_weights, dtype=float)

    if (w < 0).any() or w.sum() <= 0:
        raise ValueError(
            "Accessibility weights must be non-negative "
            "and have a positive sum."
        )

    # Normalize the selected weights to sum to 1
    w = w / w.sum()

    # Weighted accessibility score
    d["access_weighted"] = (
        comp[w.index] * w
    ).sum(axis=1, min_count=len(w))

    return d


# --------------------------------------------------
# 3. Amenity density features
# --------------------------------------------------

def fit_density_params(train, count_bases):
    params = {}

    for base in count_bases:
        for radius in ["1km", "2km"]:
            col = f"n_{base}_{radius}"

            if col not in train.columns:
                continue

            params[col] = max(
                float(train[col].quantile(0.95)),
                1.0
            )

    return params

def add_density_features(d, params, count_bases):
    d = d.copy()

    for radius in ["1km", "2km"]:

        # Select available amenity-count columns
        cols = [
            f"n_{base}_{radius}"
            for base in count_bases
            if f"n_{base}_{radius}" in params
        ]

        if not cols:
            raise ValueError(
                f"No amenity-count columns available for {radius}"
            )

        # Normalize each amenity count separately
        normalized_counts = pd.DataFrame(
            {
                col: (d[col] / params[col]).clip(0, 1)
                for col in cols
            },
            index=d.index
        )

        # Average normalized counts
        d[f"amenity_density_{radius}"] = (
            normalized_counts.mean(axis=1)
        )

        # Sum original, unnormalized counts
        d[f"n_total_{radius}"] = d[cols].sum(axis=1)

    return d


# --------------------------------------------------
# 4. Accessibility binning and classification
# --------------------------------------------------

def fit_bin_edges(train, score):
    return [
        float(train[score].quantile(1 / 3)),
        float(train[score].quantile(2 / 3))
    ]


def add_access_class(d, edges, score):
    d = d.copy()

    d["access_class"] = pd.cut(
        d[score],
        bins=[-np.inf, edges[0], edges[1], np.inf],
        labels=["Remote", "Mid", "Prime"]
    ).astype("string")

    return d
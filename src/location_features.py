from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# Project paths
ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = ROOT / "data" / "processed" / "clean_v2.csv"
EXTERNAL_DIR = ROOT / "data" / "external"
OUTPUT_FILE = ROOT / "data" / "processed" / "location_features.csv"

UTM = "EPSG:32643"

AMENITY_FILES = {
    "railway": "railway.gpkg",
    "bus": "bus.gpkg",
    "school": "school.gpkg",
    "hospital": "hospital.gpkg",
    "mall": "mall.gpkg",
    "highway": "highway.gpkg",
    "mainroad": "mainroad.gpkg",
}


def load_properties():
    """Load properties and create projected point geometries."""
    df = pd.read_csv(INPUT_FILE)

    required = {"prop_id", "latitude", "longitude"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required property columns: {sorted(missing)}")

    if df["prop_id"].duplicated().any():
        raise ValueError("Duplicate prop_id values found in the property data.")

    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")

    valid = (
        df["latitude"].between(-90, 90)
        & df["longitude"].between(-180, 180)
    )

    if not valid.all():
        raise ValueError(
            f"{(~valid).sum()} properties have invalid coordinates. "
            "Resolve them before generating location features."
        )

    props = gpd.GeoDataFrame(
        df[["prop_id"]].copy(),
        geometry=gpd.points_from_xy(df["longitude"], df["latitude"]),
        crs="EPSG:4326",
    ).to_crs(UTM)

    return df, props


def load_amenities():
    """Load available amenity GeoPackages and project them to UTM."""
    amenities = {}

    for name, filename in AMENITY_FILES.items():
        path = EXTERNAL_DIR / filename

        if not path.exists():
            print(f"Skipping {name}: {filename} not found.")
            continue

        gdf = gpd.read_file(path)

        if gdf.crs is None:
            raise ValueError(f"{filename} has no CRS defined.")

        gdf = gdf.to_crs(UTM)

        gdf = gdf[
            gdf.geometry.notna() & ~gdf.geometry.is_empty
        ].copy()

        if gdf.empty:
            print(f"Skipping {name}: no valid geometries.")
            continue

        amenities[name] = gdf

        print(f"Loaded {name}: {len(gdf)} features")

    if not amenities:
        raise FileNotFoundError(
            f"No usable amenity GeoPackages found in {EXTERNAL_DIR}"
        )

    return amenities


def nearest_distance_km(props, amenity_gdf):
    """Calculate straight-line distance to the nearest amenity, in km."""
    geometries = amenity_gdf.geometry

    valid = geometries.notna() & ~geometries.is_empty
    geometries = geometries[valid]

    if geometries.empty:
        return np.full(len(props), np.nan)

    # A spatial index allows efficient nearest-neighbour queries.
    tree = geometries.sindex

    nearest = gpd.sjoin_nearest(
        props[["geometry"]],
        amenity_gdf.loc[valid, ["geometry"]],
        how="left",
        distance_col="distance_m",
    )

    # Equidistant nearest geometries may produce multiple matches.
    # Their distances are equal, so keep one result per property.
    nearest = nearest[~nearest.index.duplicated(keep="first")]

    return (
        nearest["distance_m"]
        .reindex(props.index)
        .to_numpy()
        / 1000
    )


def amenity_count_within(props, amenity_gdf, radius_m):
    """Count mapped amenities within a radius of each property."""
    geometries = amenity_gdf.geometry

    valid = geometries.notna() & ~geometries.is_empty
    geometries = geometries[valid]

    if geometries.empty:
        return np.full(len(props), np.nan)

    # Count point locations directly. Use centroids for other geometries.
    geometries = geometries.copy()
    non_points = ~geometries.geom_type.isin(["Point", "MultiPoint"])
    geometries.loc[non_points] = geometries.loc[non_points].centroid

    geometries = geometries[
        geometries.notna() & ~geometries.is_empty
    ]

    if geometries.empty:
        return np.full(len(props), np.nan)

    coords = np.column_stack([
        geometries.geometry.x,
        geometries.geometry.y,
    ])

    tree = cKDTree(coords)

    property_coords = np.column_stack([
        props.geometry.x,
        props.geometry.y,
    ])

    return np.asarray([
        len(indices)
        for indices in tree.query_ball_point(property_coords, r=radius_m)
    ])


def main():
    """Build and save property-level location features."""
    print("Loading property data...")
    df, props = load_properties()

    print(f"Properties loaded: {len(df)}")

    print("\nLoading amenity datasets...")
    amenities = load_amenities()

    # Keep the original property order and join key.
    feats = df[["prop_id"]].copy()

    print("\nCalculating nearest distances...")
    for name, gdf in amenities.items():
        col = f"dist_{name}_km"
        feats[col] = nearest_distance_km(props, gdf)
        print(f"Calculated {col}")

    print("\nCalculating amenity counts...")
    for name in ["school", "hospital", "bus", "mall"]:
        if name not in amenities:
            continue

        for radius_m in [1000, 2000]:
            col = f"n_{name}_{radius_m // 1000}km"

            feats[col] = amenity_count_within(
                props,
                amenities[name],
                radius_m,
            )

            print(f"Calculated {col}")

    # Basic validation
    if len(feats) != len(df):
        raise RuntimeError("Output row count differs from input row count.")

    if feats["prop_id"].duplicated().any():
        raise RuntimeError("Duplicate prop_id values in output.")

    if not feats["prop_id"].equals(df["prop_id"]):
        raise RuntimeError("Output property order differs from input.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(OUTPUT_FILE, index=False)

    print("\nLocation features successfully saved.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Shape: {feats.shape}")


if __name__ == "__main__":
    main()


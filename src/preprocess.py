"""
preprocess.py
─────────────────────────────────────────────────────────────────────────────
Reads from  : data/raw/data_centers/data_centers_raw.csv
              data/raw/epa_superfund/epa_superfund_raw.csv
Writes to   : data/interim/data_centers_cleaned.gpkg
              data/interim/epa_superfund_deduplicated.gpkg

Run         : python src/preprocess.py
Next step   : python src/build_features.py

Notes
─────
Both outputs are GeoDataFrames in EPSG:4326.
  dc_df  : point geometry engineered from lat/long
  sf_df  : polygon geometry parsed from geometry_wkt column

GeoPackage (.gpkg) does not support pandas nullable Int64 dtype.
Nullable integer columns are cast to float64 to preserve NaN values.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from shapely.errors import WKTReadingError
from pathlib import Path

# ── PATHS ────────────────────────────────────────────────────────────────────

RAW_DC  = Path("data/raw/data_centers/data_centers_raw.csv")
RAW_SF  = Path("data/raw/epa_superfund/superfund_raw.csv")

INTERIM_DC = Path("data/interim/data_centers_cleaned.gpkg")
INTERIM_SF = Path("data/interim/superfund_deduplicated.gpkg")

CRS     = "EPSG:4326"
EO_DATE = pd.Timestamp("2025-07-01")

# ── REGION MAP ───────────────────────────────────────────────────────────────

CENSUS_REGION_MAP = {
    "CT": "Northeast", "ME": "Northeast", "MA": "Northeast", "NH": "Northeast",
    "RI": "Northeast", "VT": "Northeast", "NJ": "Northeast", "NY": "Northeast",
    "PA": "Northeast",
    "IL": "Midwest",   "IN": "Midwest",   "MI": "Midwest",   "OH": "Midwest",
    "WI": "Midwest",   "IA": "Midwest",   "KS": "Midwest",   "MN": "Midwest",
    "MO": "Midwest",   "NE": "Midwest",   "ND": "Midwest",   "SD": "Midwest",
    "DE": "South",     "FL": "South",     "GA": "South",     "MD": "South",
    "NC": "South",     "SC": "South",     "VA": "South",     "DC": "South",
    "WV": "South",     "AL": "South",     "KY": "South",     "MS": "South",
    "TN": "South",     "AR": "South",     "LA": "South",     "OK": "South",
    "TX": "South",
    "AZ": "West",      "CO": "West",      "ID": "West",      "MT": "West",
    "NV": "West",      "NM": "West",      "UT": "West",      "WY": "West",
    "AK": "West",      "CA": "West",      "HI": "West",      "OR": "West",
    "WA": "West",
}

# ── SF FEATURE TYPE PRIORITY for deduplication ───────────────────────────────

FEATURE_TYPE_PRIORITY = [
    "Comprehensive Site Area",
    "Comprehensive Site Boundary",
    "Total Site Polygon/OU Aggregation",
    "Total Site Polygon/OU Boundary Aggregation",
    "Current Ground Boundary",
    "Site Boundary",
    "OU Boundary Aggregation",
    "OU Boundary",
    "Extent of Contamination",
    "Contamination Boundary",
    "Contamination Boundary (Groundwater)",
    "Waste in Place",
    "Other",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def parse_wkt_safe(geom_str):
    """Parse a WKT string to a Shapely geometry, returning None on failure."""
    try:
        return wkt.loads(geom_str)
    except (WKTReadingError, TypeError, Exception):
        return None


def nullable_int_to_float(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Cast nullable Int64 columns to float64 for GeoPackage compatibility.
    NaN values are preserved. Document this in calling function.
    """
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    return df

def normalize_gis_area(row):
    area = row['GIS_AREA']
    units = str(row['GIS_AREA_UNITS']).strip().lower()
    if units in ["acres", "acre"]:
        return area
    elif units == "square miles":
        return area * 640
    elif units == "miles":
        return None
    else:
        return area



# ─────────────────────────────────────────────────────────────────────────────
# DATA CENTER CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_data_centers(path: Path) -> gpd.GeoDataFrame:
    """
    Reads raw data center CSV, fixes dtypes, engineers simple derived columns,
    and creates point geometry from lat/long.

    Geometry  : Point from lat/long → EPSG:4326
    Output    : data/interim/data_centers_cleaned.gpkg

    Dtype fixes
    ───────────
    Dates     : date_created, date_updated              → datetime64[ns]
    Floats    : lat, long                               → float64
                mw, facility_size_sqft,
                property_size_acres, project_cost       → float64
                number_of_generators, number_of_buildings,
                sizerank, days_since_eo                 → float64 (NaN-safe,
                                                          gpkg compat)
    Bool      : invalid_coords, post_eo                 → bool
    int8      : location_confidence_flag                → int8
    int64     : source_count                            → int64
    str       : zip                                     → str
    """

    print("\n── Data Centers ─────────────────────────────────────")
    df = pd.read_csv(path, dtype=str)
    print(f"  Loaded:  {df.shape[0]:,} rows × {df.shape[1]} cols")

    # ── Dates ────────────────────────────────────────────────────────────────
    for col in ["date_created", "date_updated"]:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y", errors="coerce")

    n_bad_dates = df["date_created"].isnull().sum()
    if n_bad_dates:
        print(f"  ⚠  date_created parse failures: {n_bad_dates}")

    # ── Coordinates ──────────────────────────────────────────────────────────
    df["lat"]  = pd.to_numeric(df["lat"],  errors="coerce")
    df["long"] = pd.to_numeric(df["long"], errors="coerce")

    invalid_coords = (
        df["lat"].isnull()  | df["long"].isnull()  |
        (df["lat"]  <  24)  | (df["lat"]  >  50)   |
        (df["long"] < -125) | (df["long"] > -66)
    )
    df["invalid_coords"] = invalid_coords.astype(bool)

    n_invalid = invalid_coords.sum()
    if n_invalid:
        print(f"  ⚠  Rows with invalid/missing coordinates: {n_invalid}")

    # ── Physical measurements → float64 ──────────────────────────────────────
    for col in ["mw", "facility_size_sqft", "property_size_acres", "project_cost"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Nullable integers → float64 for gpkg compat ───────────────────────────
    for col in ["number_of_generators", "number_of_buildings", "sizerank"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # ── Status ───────────────────────────────────────────────────────────────
    df["status"] = df["status"].str.strip().str.title()
    print(f"  Status values: {df['status'].value_counts().to_dict()}")

    # ── Zip — keep as string to preserve leading zeros ────────────────────────
    df["zip"] = df["zip"].str.strip().str.zfill(5)

    # ── Source columns — drop fully empty, then count ─────────────────────────
    source_cols = [c for c in df.columns if c.startswith("info_source_")]
    empty_sources = [
        c for c in source_cols
        if df[c].str.strip().eq("").all() or df[c].isnull().all()
    ]
    if empty_sources:
        df.drop(columns=empty_sources, inplace=True)
        source_cols = [c for c in df.columns if c.startswith("info_source_")]
        print(f"  Dropped {len(empty_sources)} fully empty info_source columns")

    df["source_count"] = (
        df[source_cols]
        .apply(
            lambda row: (row.notna() & (row.astype(str).str.strip() != "")).sum(),
            axis=1
        )
        .astype("int64")
    )


    # ── Derived columns ───────────────────────────────────────────────────────
    df["census_region"] = (
        df["state"].str.strip().str.upper().map(CENSUS_REGION_MAP)
    )
    df["post_eo"] = (df["date_created"] >= EO_DATE).astype(bool)

    # days_since_eo → float64 for gpkg compat (NaT rows produce NaN)
    df["days_since_eo"] = (
        (df["date_created"] - EO_DATE).dt.days.astype("float64")
    )
    df["location_confidence_flag"] = (
        df["location_confidence"].str.strip().str.title().eq("High")
    ).astype("int8")

    n_unmapped = df["census_region"].isnull().sum()
    if n_unmapped:
        print(f"  ⚠  Rows with unmapped census_region: {n_unmapped} "
              f"(states: {df.loc[df['census_region'].isnull(), 'state'].unique()})")

    # ── Create point geometry from lat/long ───────────────────────────────────
    # Rows with invalid coords get a null geometry — not dropped
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["long"], df["lat"]),
        crs=CRS,
    )

    n_null_geom = gdf.geometry.isnull().sum()
    if n_null_geom:
        print(f"  ⚠  Rows with null geometry: {n_null_geom}")

    print(f"  Output:  {gdf.shape[0]:,} rows × {gdf.shape[1]} cols | CRS: {gdf.crs}")
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# SUPERFUND CLEANING + DEDUPLICATION
# ─────────────────────────────────────────────────────────────────────────────

def clean_superfund(path: Path) -> gpd.GeoDataFrame:
    """
    Reads raw superfund CSV, fixes dtypes, deduplicates to one row per EPA_ID,
    and parses geometry_wkt into a real geometry column.

    Geometry  : Polygon/MultiPolygon from geometry_wkt → EPSG:4326
    Output    : data/interim/epa_superfund_deduplicated.gpkg

    Deduplication strategy
    ──────────────────────
    Priority    : SITE_FEATURE_TYPE in FEATURE_TYPE_PRIORITY order
    Tiebreaker  : largest GIS_AREA

    Dtype fixes
    ───────────
    Floats    : GIS_AREA, Shape__Area, Shape__Length    → float64
                OBJECTID, REGION_CODE                   → float64 (gpkg compat)
    int8      : npl_final                               → int8
    """

    print("\n── EPA Superfund ─────────────────────────────────────")
    df = pd.read_csv(path, dtype=str)
    print(f"  Loaded:  {df.shape[0]:,} rows × {df.shape[1]} cols")
    print(f"  Unique EPA_IDs (sites): {df['EPA_ID'].nunique():,}")

    # ── Numeric columns ───────────────────────────────────────────────────────
    for col in ["GIS_AREA", "Shape__Area", "Shape__Length"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["GIS_AREA_ACRES"] = df.apply(normalize_gis_area, axis=1)
    df["linear_site"] = (
            df["GIS_AREA_UNITS"].str.strip().str.lower() == "miles"
    ).astype("int8")

    n_linear = df["linear_site"].sum()
    if n_linear:
        print(f"  ⚠  Linear sites (river/stream, non-area): {n_linear}")

    # OBJECTID and REGION_CODE → float64 for gpkg compat
    for col in ["OBJECTID", "REGION_CODE"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    n_bad_area = df["GIS_AREA"].isnull().sum()
    if n_bad_area:
        print(f"  ⚠  GIS_AREA parse failures: {n_bad_area}")

    # ── NPL status flag ───────────────────────────────────────────────────────
    df["npl_final"] = (
        df["NPL_STATUS_CODE"].str.strip().eq("F")
    ).astype("int8")

    print(f"  NPL_STATUS_CODE: {df['NPL_STATUS_CODE'].value_counts().to_dict()}")
    print(f"  SITE_FEATURE_TYPE: {df['SITE_FEATURE_TYPE'].value_counts().to_dict()}")

    # ── Deduplication ─────────────────────────────────────────────────────────
    priority_map = {ft: i for i, ft in enumerate(FEATURE_TYPE_PRIORITY)}
    df["_feature_priority"] = (
        df["SITE_FEATURE_TYPE"].map(priority_map).fillna(len(FEATURE_TYPE_PRIORITY))
    )

    df_sorted = df.sort_values(
        by=["EPA_ID", "_feature_priority", "GIS_AREA"],
        ascending=[True, True, False],
        na_position="last",
    )
    df_dedup = df_sorted.drop_duplicates(subset="EPA_ID", keep="first").copy()
    df_dedup.drop(columns=["_feature_priority"], inplace=True)

    n_dropped = len(df) - len(df_dedup)
    print(f"  Deduplicated: {len(df):,} → {len(df_dedup):,} rows "
          f"({n_dropped:,} duplicate feature rows removed)")

    # ── Parse geometry_wkt → Shapely geometry ────────────────────────────────
    df_dedup["geometry"] = df_dedup["geometry_wkt"].apply(parse_wkt_safe)

    n_null_geom = df_dedup["geometry"].isnull().sum()
    n_bad_wkt   = df_dedup["geometry_wkt"].notnull().sum() - df_dedup["geometry"].notnull().sum()
    if n_null_geom:
        print(f"  ⚠  Rows with null geometry: {n_null_geom}")
    if n_bad_wkt:
        print(f"  ⚠  WKT strings that failed to parse: {n_bad_wkt}")

    # ── Create GeoDataFrame ───────────────────────────────────────────────────
    gdf = gpd.GeoDataFrame(df_dedup, geometry="geometry", crs=CRS)

    print(f"  Final NPL sites (npl_final=1): {gdf['npl_final'].sum():,}")
    print(f"  Output:  {gdf.shape[0]:,} rows × {gdf.shape[1]} cols | CRS: {gdf.crs}")
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  preprocess.py — raw → interim")
    print("=" * 55)

    # ── Ensure output directory exists ────────────────────────────────────────
    INTERIM_DC.parent.mkdir(parents=True, exist_ok=True)

    # ── Process ───────────────────────────────────────────────────────────────
    dc_clean = clean_data_centers(RAW_DC)
    sf_clean = clean_superfund(RAW_SF)

    # ── Write GeoPackage — preserves dtypes + geometry natively ───────────────
    dc_clean.to_file(INTERIM_DC, driver="GPKG")
    sf_clean.to_file(INTERIM_SF, driver="GPKG")

    print(f"\n  ✓ Written: {INTERIM_DC}")
    print(f"  ✓ Written: {INTERIM_SF}")

    print("\n" + "=" * 55)
    print("  Done. Next: dvc add data/interim/ && git commit")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
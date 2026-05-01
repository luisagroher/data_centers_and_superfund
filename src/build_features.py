"""
build_features.py
─────────────────────────────────────────────────────────────────────────────
Reads from  : data/interim/data_centers_cleaned.gpkg
              data/interim/epa_superfund_deduplicated.gpkg
Writes to   : data/processed/dc_features.gpkg
              data/processed/sf_features.gpkg

Run         : python src/build_features.py

Notes
─────
DC-centric  : distance measured to nearest SF site boundary (polygon edge)
              using sjoin_nearest — most defensible for exposure assessment
SF-centric  : DC counts measured from SF site centroid using buffer
              simpler and consistent across sites of varying size

All distance calculations performed in EPSG:5070 (US Albers Equal Area).
Outputs reprojected to EPSG:4326 for mapping compatibility.
─────────────────────────────────────────────────────────────────────────────
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path

# ── PATHS ────────────────────────────────────────────────────────────────────

INTERIM_DC  = Path("data/interim/data_centers_cleaned.gpkg")
INTERIM_SF  = Path("data/interim/superfund_deduplicated.gpkg")

PROCESSED_DC = Path("data/processed/data_center_features.gpkg")
PROCESSED_SF = Path("data/processed/superfund_features.gpkg")

# ── CONSTANTS ────────────────────────────────────────────────────────────────

CRS_METRIC          = "EPSG:5070"   # US Albers Equal Area — distance calcs
CRS_WGS84           = "EPSG:4326"   # output CRS for mapping
METERS_PER_MILE     = 1609.34
DISTANCE_THRESHOLDS = [0.5, 1, 3, 5]  # miles
NPL_FINAL_ONLY      = True            # toggle for sensitivity analysis

# ── STATUS GROUP MAP ─────────────────────────────────────────────────────────

STATUS_GROUP_MAP = {
    "Proposed"                             : "In Pipeline",
    "Approved/Permitted/Under Construction": "In Pipeline",
    "Expanding"                            : "In Pipeline",
    "Operating"                            : "Operating",
    "Cancelled"                            : "Inactive",
    "Suspended"                            : "Inactive",
    "Unknown"                              : "Inactive",
}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — READ + VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def read_and_validate(interim_dc: Path, interim_sf: Path):
    """
    Reads both interim GeoDataFrames, validates CRS, filters invalid rows,
    and adds status_group to dc_gdf.
    """
    print("\n── Step 1: Read + Validate ───────────────────────────")

    dc = gpd.read_file(interim_dc)
    sf = gpd.read_file(interim_sf)

    print(f"  DC loaded:  {len(dc):,} rows | CRS: {dc.crs}")
    print(f"  SF loaded:  {len(sf):,} rows | CRS: {sf.crs}")

    # ── CRS validation ────────────────────────────────────────────────────────
    assert dc.crs.to_epsg() == 4326, f"DC CRS mismatch: {dc.crs}"
    assert sf.crs.to_epsg() == 4326, f"SF CRS mismatch: {sf.crs}"
    print("  ✓ CRS validated: both EPSG:4326")

    # ── Filter invalid DC coordinates ─────────────────────────────────────────
    n_before = len(dc)
    dc = dc[~dc["invalid_coords"]].copy()
    print(f"  DC after coord filter: {len(dc):,} (removed {n_before - len(dc)})")

    # ── Filter SF to Final NPL only ───────────────────────────────────────────
    n_before = len(sf)
    if NPL_FINAL_ONLY:
        sf = sf[sf["npl_final"] == 1].copy()
        print(f"  SF after NPL filter (Final only): {len(sf):,} "
              f"(removed {n_before - len(sf)} non-final sites)")
    else:
        print(f"  SF NPL filter skipped (NPL_FINAL_ONLY=False): {len(sf):,} sites")

    # ── Drop geometry_wkt — geometry column exists ────────────────────────────
    if "geometry_wkt" in sf.columns:
        sf = sf.drop(columns=["geometry_wkt"])
        print("  Dropped geometry_wkt from SF (geometry column exists)")

    # ── Filter SF to valid geometry ───────────────────────────────────────────
    n_before = len(sf)
    sf = sf[sf.geometry.notnull()].copy()
    if len(sf) < n_before:
        print(f"  ⚠  SF rows with null geometry dropped: {n_before - len(sf)}")

    # ── Add status_group to DC ────────────────────────────────────────────────
    dc["status_group"] = dc["status"].map(STATUS_GROUP_MAP)
    n_unmapped = dc["status_group"].isnull().sum()
    if n_unmapped:
        print(f"  ⚠  Unmapped status values: {n_unmapped} "
              f"({dc.loc[dc['status_group'].isnull(), 'status'].unique()})")

    print(f"  DC status_group: {dc['status_group'].value_counts().to_dict()}")

    return dc, sf


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — REPROJECT TO EPSG:5070
# ─────────────────────────────────────────────────────────────────────────────

def reproject(dc: gpd.GeoDataFrame, sf: gpd.GeoDataFrame):
    """Reproject both GeoDataFrames to EPSG:5070 for metric distance calcs."""
    print("\n── Step 2: Reproject to EPSG:5070 ───────────────────")

    dc_metric = dc.to_crs(CRS_METRIC)
    sf_metric = sf.to_crs(CRS_METRIC)

    print(f"  ✓ DC reprojected: {dc_metric.crs}")
    print(f"  ✓ SF reprojected: {sf_metric.crs}")

    return dc_metric, sf_metric


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — DC-CENTRIC SPATIAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_dc_features(
    dc_metric: gpd.GeoDataFrame,
    sf_metric: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    For each DC, computes:
    - dist_to_nearest_sf_m / _mi : distance to nearest SF boundary
    - nearest_sf_name             : name of nearest SF site
    - nearest_sf_npl_status       : NPL status of nearest SF site
    - nearest_sf_gis_area_acres   : area of nearest SF site
    - near_sf_{X}mi               : binary flag per threshold
    - n_sf_within_{X}mi           : count of SF sites within threshold
    """
    print("\n── Step 3: DC-Centric Spatial Features ──────────────")

    # ── Nearest SF site per DC (distance to polygon boundary) ────────────────
    print("  Running sjoin_nearest (DC → SF boundary)...")
    nearest = gpd.sjoin_nearest(
        dc_metric[["geometry", "status_group"]],
        sf_metric[["geometry", "SITE_NAME", "NPL_STATUS_CODE", "GIS_AREA_ACRES"]],
        how="left",
        distance_col="dist_to_nearest_sf_m",
    )

    # sjoin_nearest may produce duplicates if equidistant — keep closest
    nearest = nearest[~nearest.index.duplicated(keep="first")]

    # ── Attach distance + nearest SF attributes to dc_metric ─────────────────
    dc_metric = dc_metric.copy()
    dc_metric["dist_to_nearest_sf_m"]    = nearest["dist_to_nearest_sf_m"]
    dc_metric["dist_to_nearest_sf_mi"]   = (
        nearest["dist_to_nearest_sf_m"] / METERS_PER_MILE
    )
    dc_metric["nearest_sf_name"]         = nearest["SITE_NAME"]
    dc_metric["nearest_sf_npl_status"]   = nearest["NPL_STATUS_CODE"]
    dc_metric["nearest_sf_gis_area_acres"] = nearest["GIS_AREA_ACRES"]

    print(f"  ✓ Nearest SF computed for {dc_metric['dist_to_nearest_sf_mi'].notnull().sum():,} DCs")
    print(f"  Distance to nearest SF (miles):")
    print(f"    median: {dc_metric['dist_to_nearest_sf_mi'].median():.2f}")
    print(f"    mean:   {dc_metric['dist_to_nearest_sf_mi'].mean():.2f}")
    print(f"    min:    {dc_metric['dist_to_nearest_sf_mi'].min():.4f}")
    print(f"    max:    {dc_metric['dist_to_nearest_sf_mi'].max():.2f}")

    # ── Binary flags per threshold ────────────────────────────────────────────
    for mi in DISTANCE_THRESHOLDS:
        col = f"near_sf_{str(mi).replace('.', '_')}mi"
        dc_metric[col] = (
            dc_metric["dist_to_nearest_sf_mi"] <= mi
        ).astype("int8")
        n = dc_metric[col].sum()
        print(f"  near_sf_{mi}mi: {n:,} DCs ({n/len(dc_metric)*100:.1f}%)")

    # ── Count of SF sites within each threshold ───────────────────────────────
    print("  Computing SF site counts within thresholds per DC...")
    for mi in DISTANCE_THRESHOLDS:
        meters = mi * METERS_PER_MILE
        col    = f"n_sf_within_{str(mi).replace('.', '_')}mi"
        counts = []
        for geom in dc_metric.geometry:
            buffer  = geom.buffer(meters)
            n_sites = sf_metric.geometry.intersects(buffer).sum()
            counts.append(n_sites)
        dc_metric[col] = counts
        print(f"  n_sf_within_{mi}mi: mean={np.mean(counts):.2f} "
              f"max={np.max(counts)}")

    return dc_metric


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SF-CENTRIC SPATIAL FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def build_sf_features(
    sf_metric: gpd.GeoDataFrame,
    dc_metric: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    For each SF site centroid, computes:
    - dc_count_within_{X}mi       : total DCs within threshold
    - pipeline_count_within_{X}mi : In Pipeline DCs within threshold
    - operating_count_within_{X}mi: Operating DCs within threshold
    """
    print("\n── Step 4: SF-Centric Spatial Features ──────────────")

    sf_metric = sf_metric.copy()

    # ── Centroid for buffer operations ────────────────────────────────────────
    sf_metric["centroid"] = sf_metric.geometry.centroid
    print(f"  ✓ Centroids computed for {len(sf_metric):,} SF sites")

    # ── Subsets by status group ───────────────────────────────────────────────
    dc_pipeline  = dc_metric[dc_metric["status_group"] == "In Pipeline"]
    dc_operating = dc_metric[dc_metric["status_group"] == "Operating"]

    for mi in DISTANCE_THRESHOLDS:
        meters = mi * METERS_PER_MILE

        total_counts     = []
        pipeline_counts  = []
        operating_counts = []

        for centroid in sf_metric["centroid"]:
            buffer = centroid.buffer(meters)
            total_counts.append(
                dc_metric.geometry.intersects(buffer).sum()
            )
            pipeline_counts.append(
                dc_pipeline.geometry.intersects(buffer).sum()
            )
            operating_counts.append(
                dc_operating.geometry.intersects(buffer).sum()
            )

        mi_str = str(mi).replace(".", "_")
        sf_metric[f"dc_count_within_{mi_str}mi"]       = total_counts
        sf_metric[f"pipeline_count_within_{mi_str}mi"] = pipeline_counts
        sf_metric[f"operating_count_within_{mi_str}mi"]= operating_counts

        n_with_dc = sum(c > 0 for c in total_counts)
        print(f"  {mi}mi: {n_with_dc:,} SF sites have ≥1 DC nearby "
              f"({n_with_dc/len(sf_metric)*100:.1f}%)")

    # ── Drop centroid helper column before saving ─────────────────────────────
    sf_metric = sf_metric.drop(columns=["centroid"])

    return sf_metric


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — REPROJECT BACK TO WGS84
# ─────────────────────────────────────────────────────────────────────────────

def reproject_output(
    dc_metric: gpd.GeoDataFrame,
    sf_metric: gpd.GeoDataFrame,
):
    """Reproject both outputs back to EPSG:4326 for mapping compatibility."""
    print("\n── Step 5: Reproject outputs to EPSG:4326 ────────────")

    dc_out = dc_metric.to_crs(CRS_WGS84)
    sf_out = sf_metric.to_crs(CRS_WGS84)

    print(f"  ✓ DC output CRS: {dc_out.crs}")
    print(f"  ✓ SF output CRS: {sf_out.crs}")

    return dc_out, sf_out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — WRITE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def write_outputs(
    dc_out: gpd.GeoDataFrame,
    sf_out: gpd.GeoDataFrame,
):
    """Write both GeoDataFrames to processed/ as GeoPackage."""
    print("\n── Step 6: Write Outputs ─────────────────────────────")

    PROCESSED_DC.parent.mkdir(parents=True, exist_ok=True)

    dc_out.to_file(PROCESSED_DC, driver="GPKG")
    sf_out.to_file(PROCESSED_SF, driver="GPKG")

    print(f"  ✓ Written: {PROCESSED_DC}")
    print(f"    {len(dc_out):,} rows × {len(dc_out.columns)} cols")

    print(f"  ✓ Written: {PROCESSED_SF}")
    print(f"    {len(sf_out):,} rows × {len(sf_out.columns)} cols")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  build_features.py — interim → processed")
    print(f"  NPL_FINAL_ONLY = {NPL_FINAL_ONLY}")
    print(f"  Distance thresholds (mi): {DISTANCE_THRESHOLDS}")
    print("=" * 55)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    dc, sf                   = read_and_validate(INTERIM_DC, INTERIM_SF)
    dc_metric, sf_metric     = reproject(dc, sf)
    dc_metric                = build_dc_features(dc_metric, sf_metric)
    sf_metric                = build_sf_features(sf_metric, dc_metric)
    dc_out, sf_out           = reproject_output(dc_metric, sf_metric)
    write_outputs(dc_out, sf_out)

    print("\n" + "=" * 55)
    print("  Done. Next: dvc add data/processed/ && git commit")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
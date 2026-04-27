"""
fetch_data.py
-------------
Fetches:
  1. Data center locations from a public Google Sheet (CSV export)
  2. EPA Superfund sites from an ArcGIS Feature Service

Outputs land in data/raw/ and are DVC-ready.
Run from the project root: python src/fetch_data.py
"""

import os
import json
import time
import requests
import pandas as pd
import geopandas as gpd
from pathlib import Path

# ── Output dirs ──────────────────────────────────────────────
RAW = Path("data/raw")
DC_DIR   = RAW / "data_centers"
EPA_DIR  = RAW / "epa_superfund"
DC_DIR.mkdir(parents=True, exist_ok=True)
EPA_DIR.mkdir(parents=True, exist_ok=True)

# ── Shared session ───────────────────────────────────────────
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "datacenter-superfund-research/1.0"})


# ════════════════════════════════════════════════════════════
# 1. GOOGLE SHEET → DATA CENTERS
# ════════════════════════════════════════════════════════════

SHEET_ID = "1JJ6kcVo-NjlAYtznwHOki2DVl4WWV6lhy-eXhFCdKKU"
SHEET_GID = "386766486"
SHEET_CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}"
    f"/export?format=csv&gid={SHEET_GID}"
)

def fetch_google_sheet() -> pd.DataFrame:
    """
    Exports the Google Sheet tab as CSV.
    Sheet must be publicly accessible (File → Share → Anyone with the link).
    """
    print("📥  Fetching data center Google Sheet …")
    resp = SESSION.get(SHEET_CSV_URL, timeout=30)
    resp.raise_for_status()

    # Google may redirect to a login page if the sheet is private
    if "accounts.google.com" in resp.url:
        raise PermissionError(
            "Sheet is not publicly accessible. "
            "Set share to 'Anyone with the link → Viewer' and retry."
        )

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    print(f"   ✅  {len(df):,} rows, columns: {list(df.columns)}")
    return df


def save_data_centers(df: pd.DataFrame):
    out_csv = DC_DIR / "data_centers_raw.csv"
    df.to_csv(out_csv, index=False)
    print(f"   💾  Saved → {out_csv}")

    # Try to make a GeoJSON if lat/lon columns exist
    lat_candidates = [c for c in df.columns if "lat" in c.lower()]
    lon_candidates = [c for c in df.columns if "lon" in c.lower() or "lng" in c.lower()]

    if lat_candidates and lon_candidates:
        lat_col, lon_col = lat_candidates[0], lon_candidates[0]
        geo_df = df.dropna(subset=[lat_col, lon_col]).copy()
        gdf = gpd.GeoDataFrame(
            geo_df,
            geometry=gpd.points_from_xy(geo_df[lon_col], geo_df[lat_col]),
            crs="EPSG:4326",
        )
        out_geo = DC_DIR / "data_centers_raw.geojson"
        gdf.to_file(out_geo, driver="GeoJSON")
        print(f"   🗺   GeoJSON saved → {out_geo}  ({len(gdf):,} geocoded rows)")
    else:
        print("   ⚠️   No lat/lon columns detected — skipping GeoJSON.")
        print(f"        Detected columns: {list(df.columns)}")


# ════════════════════════════════════════════════════════════
# 2. ARCGIS FEATURE SERVICE → EPA SUPERFUND
# ════════════════════════════════════════════════════════════

ARCGIS_ITEM_ID = "d6e1591d9a424f1fa6d95a02095a06d7"
ARCGIS_ITEM_URL = (
    f"https://www.arcgis.com/sharing/rest/content/items"
    f"/{ARCGIS_ITEM_ID}?f=json"
)

def get_arcgis_service_url() -> str:
    """
    Resolves the underlying FeatureServer REST URL from an ArcGIS item ID.
    """
    print("🔍  Resolving ArcGIS service URL …")
    resp = SESSION.get(ARCGIS_ITEM_URL, timeout=30)
    resp.raise_for_status()
    meta = resp.json()

    service_url = meta.get("url", "")
    if not service_url:
        raise ValueError(
            f"Could not resolve service URL from item metadata.\n"
            f"Full metadata: {json.dumps(meta, indent=2)}"
        )

    # Ensure we're pointing at layer 0 of the FeatureServer
    if not service_url.endswith("/0"):
        service_url = service_url.rstrip("/") + "/0"

    print(f"   ✅  Service URL: {service_url}")
    return service_url


def fetch_arcgis_layer(service_url: str, page_size: int = 1000) -> gpd.GeoDataFrame:
    """
    Pages through an ArcGIS FeatureServer layer and returns a GeoDataFrame.
    ArcGIS caps results per request (usually 1000), so we paginate via
    resultOffset until no more features are returned.
    """
    query_url = f"{service_url}/query"
    params = {
        "where": "1=1",
        "outFields": "*",
        "f": "geojson",
        "resultRecordCount": page_size,
        "resultOffset": 0,
    }

    all_features = []
    page = 0

    while True:
        params["resultOffset"] = page * page_size
        print(f"   📄  Page {page + 1}  (offset {params['resultOffset']}) …", end=" ")

        resp = SESSION.get(query_url, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        features = data.get("features", [])
        print(f"{len(features)} features")

        if not features:
            break

        all_features.extend(features)
        page += 1
        time.sleep(0.3)  # be polite to the server

        # Some services don't support offset pagination; check for the flag
        if not data.get("exceededTransferLimit", False) and len(features) < page_size:
            break

    print(f"   ✅  Total features fetched: {len(all_features):,}")

    # Reconstruct as a proper GeoJSON FeatureCollection for GeoPandas
    feature_collection = {
        "type": "FeatureCollection",
        "features": all_features,
    }
    gdf = gpd.GeoDataFrame.from_features(feature_collection, crs="EPSG:4326")
    return gdf


def save_superfund(gdf: gpd.GeoDataFrame):
    out_geo = EPA_DIR / "superfund_raw.geojson"
    gdf.to_file(out_geo, driver="GeoJSON")
    print(f"   💾  GeoJSON saved → {out_geo}")

    # Also save a flat CSV (geometry as WKT) for easy inspection
    gdf_csv = gdf.copy()
    gdf_csv["geometry_wkt"] = gdf_csv.geometry.apply(
        lambda g: g.wkt if g is not None else None
    )
    gdf_csv.drop(columns="geometry").to_csv(
        EPA_DIR / "superfund_raw.csv", index=False
    )
    print(f"   💾  CSV saved  → {EPA_DIR / 'superfund_raw.csv'}")
    print(f"   📋  Columns: {list(gdf.columns)}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 55)
    print("  Data Center × Superfund — Raw Data Fetch")
    print("=" * 55 + "\n")

    # 1. Data centers
    try:
        dc_df = fetch_google_sheet()
        save_data_centers(dc_df)
    except Exception as e:
        print(f"   ❌  Data center fetch failed: {e}")

    print()

    # 2. Superfund
    try:
        service_url = get_arcgis_service_url()
        superfund_gdf = fetch_arcgis_layer(service_url)
        save_superfund(superfund_gdf)
    except Exception as e:
        print(f"   ❌  Superfund fetch failed: {e}")

    print("\n" + "=" * 55)
    print("  Done. Next: dvc add data/raw/ && git commit")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()

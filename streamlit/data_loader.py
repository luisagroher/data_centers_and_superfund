"""
data_loader.py
─────────────────────────────────────────────────────────────────────────────
Cached loaders for the two processed GeoPackages.

The data center loader applies ``DASHBOARD_STATUS_MAP`` so that every
component sees a consistent ``dashboard_status`` column.
─────────────────────────────────────────────────────────────────────────────
"""
import geopandas as gpd
import streamlit as st

from constants import DASHBOARD_STATUS_MAP, DC_PATH, SF_PATH


@st.cache_data(show_spinner="Loading data centers…")
def load_data_centers() -> gpd.GeoDataFrame:
    """
    Load the processed data center GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        Point geometries in EPSG:4326 with a ``dashboard_status`` column
        added from the raw ``status`` column.
    """
    gdf = gpd.read_file(DC_PATH)
    gdf["dashboard_status"] = gdf["status"].map(DASHBOARD_STATUS_MAP)
    return gdf


@st.cache_data(show_spinner="Loading superfund sites…")
def load_superfund_sites() -> gpd.GeoDataFrame:
    """
    Load the processed Superfund site GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        Polygon geometries in EPSG:4326. ``STATE_CODE`` is stripped of
        surrounding whitespace.
    """
    gdf = gpd.read_file(SF_PATH)
    gdf["STATE_CODE"] = gdf["STATE_CODE"].str.strip()
    return gdf

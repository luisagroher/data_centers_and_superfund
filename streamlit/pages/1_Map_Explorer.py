"""
1_Map_Explorer.py
─────────────────────────────────────────────────────────────────────────────
Explore — the interactive drill-down map: US -> state -> (city | Superfund
site). Not a narrated story; this is the exploration tool that sits ahead
of the narrative pages.

See components/map_view.py for the merged map logic (formerly
us_overview_map.py + state_site_map.py).
─────────────────────────────────────────────────────────────────────────────
"""
import sys
from pathlib import Path

STREAMLIT_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT  = STREAMLIT_DIR.parent
for _p in (STREAMLIT_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st

from components.map_view import render as render_map_view
from components.status_summary_header import render as render_status_summary
from data_loader import load_data_centers, load_superfund_sites
from src.tufte_style import define_plot_style

st.set_page_config(page_title="Map Explorer", layout="wide")
define_plot_style()

dc_gdf = load_data_centers()
sf_gdf = load_superfund_sites()

st.title("Map Explorer")
st.markdown(
    "Explore data centers and Superfund sites nationally, then drill into a "
    "state, city, or specific Superfund site."
)

render_status_summary(dc_gdf)
render_map_view(sf_gdf, dc_gdf)

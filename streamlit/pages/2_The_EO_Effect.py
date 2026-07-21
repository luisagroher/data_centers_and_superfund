"""
2_The_EO_Effect.py
─────────────────────────────────────────────────────────────────────────────
Story 2 — "The EO Effect": did AI data center proposals shift closer to
Superfund sites after Executive Order 14318 (July 2025), when did that
shift happen, and where did it concentrate.

Preliminary page, composed entirely from existing components per the
layout agreed in planning.md. The headline KPI strip, proximity-rate-by-
threshold chart, regional proximity-shift table, and state-level
SF-proximity ranking called for in planning.md are new components not
yet built, and are not included here.
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

from components.cumulative_proposals import render as render_cumulative
from components.dc_near_sf_thresholds import render as render_dc_thresholds
from components.eo_rate_by_region import render as render_eo_rate
from components.monthly_proposals_near_sf import render as render_monthly_near_sf
from components.mw_near_sf import render as render_mw_near_sf
from components.proposals_over_time import render as render_proposals_over_time
from components.sf_site_labels import render as render_sf_site_labels
from components.top_sf_by_nearby_dc import render as render_top_sf
from data_loader import load_data_centers, load_superfund_sites
from src.tufte_style import define_plot_style

st.set_page_config(page_title="The EO Effect", layout="wide")
define_plot_style()

dc_gdf = load_data_centers()
sf_gdf = load_superfund_sites()

st.title("The EO Effect")
st.markdown(
    "Did AI data center proposals shift closer to Superfund sites after the "
    "July 2025 executive order accelerating federal permitting for data "
    "center infrastructure — and if so, when did that shift happen, and "
    "where did it concentrate?"
)

st.divider()
st.header("The boom in context")
render_proposals_over_time(dc_gdf)
render_cumulative(dc_gdf)

st.divider()
st.header("Proximity to contamination")
col1, col2 = st.columns(2)
with col1:
    render_dc_thresholds(dc_gdf)
with col2:
    render_mw_near_sf(dc_gdf)

st.divider()
st.header("Did proximity shift after the EO?")
render_monthly_near_sf(dc_gdf)
render_eo_rate(dc_gdf)

st.divider()
st.header("Where it's concentrated")
render_top_sf(sf_gdf)
render_sf_site_labels(sf_gdf)

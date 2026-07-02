"""
app.py
─────────────────────────────────────────────────────────────────────────────
Streamlit entrypoint for the Data Centers × Superfund dashboard.

Sidebar
───────
- View selector  (which component to preview)
- View-specific controls (e.g. site picker + radius for "Site map")

Main area
─────────
- Renders the selected component. This is a temporary gallery layout —
  components live in ``streamlit/components/`` as independent modules and
  will be composed into the final dashboard once the team finalizes
  feedback on the candidate charts.

Run (from project root)
───────────────────────
    streamlit run streamlit/app.py

Requires the processed GeoPackages under ``data/processed/``.
If they are missing, run ``dvc pull`` first.
─────────────────────────────────────────────────────────────────────────────
"""
import sys
from pathlib import Path

# Make ``src`` importable when launched via ``streamlit run streamlit/app.py``.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from components.cumulative_proposals import render as render_cumulative
from components.dc_by_status import render as render_dc_by_status
from components.dc_near_sf_thresholds import render as render_dc_thresholds
from components.eo_rate_by_region import render as render_eo_rate
from components.monthly_proposals_near_sf import render as render_monthly_near_sf
from components.mw_by_state import render as render_mw_by_state
from components.mw_near_sf import render as render_mw_near_sf
from components.proposals_over_time import render as render_proposals_over_time
from components.state_counts import render as render_state_counts
from components.city_dc_labels import render as render_city_dc_labels
from components.sf_site_labels import render as render_sf_site_labels
from components.state_dc_labels import render as render_state_dc_labels
from components.state_site_map import render as render_site_map
from components.us_overview_map import render as render_us_overview
from components.status_summary_header import render as render_status_summary
from components.top_cities_by_dc import render as render_top_cities
from components.top_sf_by_nearby_dc import render as render_top_sf
from components.top_states_by_status import render as render_top_states
from constants import DEFAULT_RADIUS_MI, DISTANCE_THRESHOLDS_MI
from data_loader import load_data_centers, load_superfund_sites
from src.tufte_style import define_plot_style

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Data Centers × Superfund",
    layout="wide",
)
define_plot_style()
st.title("Data Centers × Superfund")

# ── Load data ────────────────────────────────────────────────
dc_gdf = load_data_centers()
sf_gdf = load_superfund_sites()

# ── Sidebar: view selector + view-specific controls ──────────
VIEWS = [
    "Site map",
    "State counts",
    "State DC labels",
    "US Overview",
    "DCs by status",
    "DCs near Superfund",
    "Top states by status",
    "Top cities by DC",
    "City DC labels",
    "Top SF sites by nearby DCs",
    "SF site labels",
    "MW by state",
    "MW near Superfund",
    "Proposals over time",
    "Monthly proposals near SF",
    "EO rate by region",
    "Cumulative proposals",
]

with st.sidebar:
    st.header("View")
    view = st.radio("Component", VIEWS, label_visibility="collapsed")

    if view == "Site map":
        st.divider()
        st.header("Select a site")

        states_with_sites = sorted(sf_gdf["STATE_CODE"].dropna().unique().tolist())
        state = st.selectbox("State", states_with_sites)

        sites_in_state = (
            sf_gdf.loc[sf_gdf["STATE_CODE"] == state, ["EPA_ID", "SITE_NAME"]]
            .sort_values("SITE_NAME")
        )

        site_name = st.selectbox(
            "Superfund site",
            sites_in_state["SITE_NAME"].tolist(),
        )
        site_epa_id = sites_in_state.loc[
            sites_in_state["SITE_NAME"] == site_name, "EPA_ID"
        ].iloc[0]

        radius_mi = st.select_slider(
            "Radius (miles)",
            options=DISTANCE_THRESHOLDS_MI,
            value=DEFAULT_RADIUS_MI,
        )

# ── Status summary strip (top of every page) ─────────────────
render_status_summary(dc_gdf)

# ── Main area: dispatch to the selected component ────────────
if view == "US Overview":
    render_us_overview(sf_gdf, dc_gdf)
elif view == "Site map":
    render_site_map(sf_gdf, dc_gdf, site_epa_id, radius_mi=radius_mi)
elif view == "State counts":
    render_state_counts(dc_gdf, sf_gdf)
elif view == "State DC labels":
    render_state_dc_labels(dc_gdf)
elif view == "DCs by status":
    render_dc_by_status(dc_gdf)
elif view == "DCs near Superfund":
    render_dc_thresholds(dc_gdf)
elif view == "Top states by status":
    render_top_states(dc_gdf)
elif view == "Top cities by DC":
    render_top_cities(dc_gdf)
elif view == "City DC labels":
    render_city_dc_labels(dc_gdf)
elif view == "Top SF sites by nearby DCs":
    render_top_sf(sf_gdf)
elif view == "SF site labels":
    render_sf_site_labels(sf_gdf)
elif view == "MW by state":
    render_mw_by_state(dc_gdf)
elif view == "MW near Superfund":
    render_mw_near_sf(dc_gdf)
elif view == "Proposals over time":
    render_proposals_over_time(dc_gdf)
elif view == "Monthly proposals near SF":
    render_monthly_near_sf(dc_gdf)
elif view == "EO rate by region":
    render_eo_rate(dc_gdf)
elif view == "Cumulative proposals":
    render_cumulative(dc_gdf)

"""
constants.py
─────────────────────────────────────────────────────────────────────────────
Shared constants for the Streamlit app.

Holds
─────
- File paths for the processed GeoPackages
- CRS used for distance / buffer operations
- Dashboard status mapping and palette
- UI defaults (distance thresholds, default radius)

Notes
─────
DASHBOARD_STATUS_MAP and DASHBOARD_STATUS_PALETTE are defined here only.
Once the upstream pipeline is updated to expose these categories on
``status_group`` directly, this block can be removed and components can
read ``status_group`` instead of ``dashboard_status``.
─────────────────────────────────────────────────────────────────────────────
"""
import pandas as pd

from src.tufte_style import COLORS

# ── PATHS ────────────────────────────────────────────────────
_DAGSHUB_BASE = (
    "https://dagshub.com/luisagroher/data_centers_and_superfund"
    "/raw/main/data/processed"
)
DC_PATH = f"{_DAGSHUB_BASE}/data_center_features.gpkg"
SF_PATH = f"{_DAGSHUB_BASE}/superfund_features.gpkg"

# ── CRS ──────────────────────────────────────────────────────
CRS_WGS84       = "EPSG:4326"
CRS_METRIC      = "EPSG:5070"
METERS_PER_MILE = 1609.34

# ── DASHBOARD STATUS ─────────────────────────────────────────
DASHBOARD_STATUS_MAP = {
    "Proposed":                              "Proposed",
    "Approved/Permitted/Under Construction": "Under Construction",
    "Operating":                             "Operational",
    "Expanding":                             "Expanding",
    "Cancelled":                             "Inactive",
    "Suspended":                             "Inactive",
    "Unknown":                               "Inactive",
}

DASHBOARD_STATUS_PALETTE = {
    "Proposed":           COLORS["proposed"],
    "Under Construction": COLORS["post_eo"],
    "Operational":        COLORS["operating"],
    "Expanding":          COLORS["superfund"],
    "Inactive":           COLORS["neutral"],
}

SUPERFUND_COLOR = COLORS["superfund"]

# ── EXECUTIVE ORDER ──────────────────────────────────────────
EO_DATE = pd.Timestamp("2025-07-23")

# ── UI DEFAULTS ──────────────────────────────────────────────
DISTANCE_THRESHOLDS_MI = [0.5, 1, 3, 5]
DEFAULT_RADIUS_MI      = 1

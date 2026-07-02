# Dashboard Changelog

Running log of visualizations in the Streamlit app. Each component is an
independent module under `streamlit/components/` and is wired into the
gallery selector in `streamlit/app.py`.

---

## Current components

### Maps

| View label   | File                                      | Source              | Notes                                            |
| ------------ | ----------------------------------------- | ------------------- | ------------------------------------------------ |
| Site map     | `components/state_site_map.py`            | new (not in 04.ipynb) | State → site picker, radius slider, Folium/Leaflet |

### Counts & rankings

| View label                  | File                                  | Notebook source        | Status toggle                  |
| --------------------------- | ------------------------------------- | -------------------- | ------------------------------ |
| State counts                | `components/state_counts.py`          | Component 1            | —                              |
| DCs by status               | `components/dc_by_status.py`          | Component 7            | — (is the breakdown)           |
| DCs near Superfund          | `components/dc_near_sf_thresholds.py` | Component 2a           | —                              |
| Top states by status        | `components/top_states_by_status.py`  | Components 9 / 10 / 10b | `dashboard_status`            |
| Top cities by DC            | `components/top_cities_by_dc.py`      | Component 3            | `All` + `dashboard_status`     |
| Top SF sites by nearby DCs  | `components/top_sf_by_nearby_dc.py`   | Component 4            | `All` + `status_group`         |
| MW by state                 | `components/mw_by_state.py`           | Component 5            | `All` + `dashboard_status`     |
| MW near Superfund           | `components/mw_near_sf.py`            | Component 6 (+ 8 dup)  | `All` + `dashboard_status`     |

### Time series

| View label                | File                                       | Notebook source     | Notes                                            |
| ------------------------- | ------------------------------------------ | ----------------- | ------------------------------------------------ |
| Proposals over time       | `components/proposals_over_time.py`        | TS 1 + TS 1b        | Merged via `All / <status>` toggle               |
| Monthly proposals near SF | `components/monthly_proposals_near_sf.py`  | TS 2                | Radius slider (was hard-coded to 1 mi)           |
| EO rate by region         | `components/eo_rate_by_region.py`          | TS 3 + rate table   | Chart + `st.dataframe` rate table below          |
| Cumulative proposals      | `components/cumulative_proposals.py`       | TS 4                | EO annotation anchored bottom to clear legend    |

### Shared infrastructure

| File                                | Purpose                                                       |
| ----------------------------------- | ------------------------------------------------------------- |
| `app.py`                            | Entrypoint, sidebar view selector, dispatch to components     |
| `constants.py`                      | Paths, CRS, status map/palette, `EO_DATE`, UI defaults        |
| `data_loader.py`                    | Cached `@st.cache_data` loaders for the two GeoPackages       |
| `components/_ts_helpers.py`         | `draw_eo_line`, `shade_pre_post` for time series components   |

---

## Not yet built (candidates from notebook 04)

From the bottom of `notebooks/04 - Dashboard.ipynb`:

- Continental US map — DC points colored by `near_sf_1mi` over SF polygons
- Regional sub-maps — Northeast, South, Midwest, West
- Atlanta zoom
- Silicon Valley zoom
- Choropleth by state (DC count / SF count / % near SF)
- Time animation — proposals pre vs post EO

Proximity/distance charts mentioned in the notebook but not yet built:

- Distribution of distance to nearest SF site by status group
- Cumulative % of DCs within X miles — pre vs post EO
- % of DCs near SF at each threshold by region
- % of SF sites with at least 1 DC nearby by threshold
- Scatter — DC size (MW) vs distance to nearest SF site

---

## Changelog

### 2026-06-18

- **Added** `components/_ts_helpers.py` (`draw_eo_line`, `shade_pre_post`)
- **Added** `EO_DATE = pd.Timestamp("2025-07-23")` to `constants.py`
- **Added** time series batch:
  - `components/proposals_over_time.py` (TS 1 + 1b merged)
  - `components/monthly_proposals_near_sf.py` (TS 2, radius slider added)
  - `components/eo_rate_by_region.py` (TS 3 + rate table)
  - `components/cumulative_proposals.py` (TS 4)
- **Added** stacked-bar batch:
  - `components/dc_by_status.py` (Component 7)
  - `components/top_cities_by_dc.py` (Component 3)
  - `components/top_sf_by_nearby_dc.py` (Component 4)
  - `components/mw_by_state.py` (Component 5; fixed `top_cities` → `top_states` bug from notebook)
  - `components/mw_near_sf.py` (Component 6, merged with duplicate Component 8)
- **Added** `components/top_states_by_status.py` — parameterized port of
  Components 9 / 10 / 10b with a status radio toggle
- **Added** first batch of components and gallery scaffold:
  - `components/state_site_map.py` (new state → site → map view, Folium)
  - `components/state_counts.py` (Component 1)
  - `components/dc_near_sf_thresholds.py` (Component 2a)
- **Added** initial app scaffold: `app.py`, `constants.py`, `data_loader.py`
- **Added** `DASHBOARD_STATUS_MAP` and `DASHBOARD_STATUS_PALETTE` to
  `constants.py` to unify status categories across components

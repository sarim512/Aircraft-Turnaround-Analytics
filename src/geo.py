"""Airport geography and the network risk map."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config as cfg
from . import viz

TYPE_RANK = {"large_airport": 3, "medium_airport": 2, "small_airport": 1}
# BTS covers the 50 states plus US territories.
US_COUNTRIES = {"US", "PR", "VI", "GU", "AS", "MP"}


def load_airport_coords(cache_path: Path = cfg.EXTERNAL_DIR / "airports.csv",
                        url: str = cfg.AIRPORTS_URL) -> pd.DataFrame:
    """US airport coordinates from OurAirports (public domain), cached locally.

    * Filters on ``iso_country`` rather than a lat/lon bounding box: a box such as
      -130 < lon < -60 drops every Alaska and Hawaii airport and keeps Canadian and
      Mexican ones.
    * Keeps operating airports only and de-duplicates IATA codes: closed or heliport
      records that share a code would otherwise duplicate rows in the merge.
    """
    cache_path = Path(cache_path)
    if not cache_path.exists():
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)

    df = pd.read_csv(cache_path, keep_default_na=False, na_values=[""],
                     usecols=["iata_code", "name", "municipality", "latitude_deg",
                              "longitude_deg", "type", "iso_country"])
    df = df[df["type"].isin(TYPE_RANK) & df["iso_country"].isin(US_COUNTRIES)
            & df["iata_code"].str.fullmatch(r"[A-Z0-9]{3}", na=False)].copy()
    df["_rank"] = df["type"].map(TYPE_RANK)
    df = df.sort_values("_rank", ascending=False).drop_duplicates("iata_code")
    return (df.rename(columns={"iata_code": "Origin", "name": "Airport_Name",
                               "municipality": "City", "latitude_deg": "Latitude",
                               "longitude_deg": "Longitude"})
              [["Origin", "Airport_Name", "City", "Latitude", "Longitude"]]
              .reset_index(drop=True))


def airport_risk_table(df: pd.DataFrame, coords: pd.DataFrame, min_events: int = 300) -> pd.DataFrame:
    """Per-airport delay-generation rate joined to coordinates (validated, never silently empty)."""
    risk = (df.groupby("Origin", observed=True)
              .agg(Delay_Rate=("TAT_Delayed", "mean"), Mean_TAT=("TAT_Actual", "mean"),
                   Median_Sched_TAT=("TAT_Scheduled", "median"), Events=("TAT_Actual", "size"))
              .query("Events >= @min_events").reset_index())
    risk["Origin"] = risk["Origin"].astype(str)
    merged = risk.merge(coords, on="Origin", how="left", validate="one_to_one")
    missing = merged["Latitude"].isna()
    if missing.all():
        raise ValueError("No airport matched the coordinate table -- check the IATA join keys.")
    if missing.any():
        print(f"Note: {missing.sum()} airports without coordinates: "
              f"{', '.join(merged.loc[missing, 'Origin'].head(10))}")
    merged = merged[~missing].copy()
    merged["Delay_Pct"] = (merged["Delay_Rate"].astype(float) * 100).round(1)
    merged["Mean_TAT"] = merged["Mean_TAT"].astype(float).round(1)
    merged["Median_Sched_TAT"] = merged["Median_Sched_TAT"].astype(float)
    return merged


def risk_map_figure(airport_map: pd.DataFrame, year: int = cfg.YEAR):
    import plotly.express as px

    fig = px.scatter_geo(
        airport_map, lat="Latitude", lon="Longitude", size="Events", color="Delay_Pct",
        hover_name="Origin",
        hover_data={"Airport_Name": True, "City": True, "Delay_Pct": ":.1f",
                    "Mean_TAT": ":.0f", "Events": ":,", "Latitude": False, "Longitude": False},
        labels={"Delay_Pct": "Delay-generating turns (%)", "Mean_TAT": "Mean ground time (min)",
                "Events": "Turnarounds", "Airport_Name": "Airport", "City": "City"},
        color_continuous_scale=viz.PLOTLY_SEQ, size_max=36, scope="usa",
        title=(f"US airport turnaround risk, {year}<br><sup>Bubble size = turnaround volume · "
               "colour = share of turnarounds that generated a departure delay</sup>"),
    )
    fig.update_traces(marker=dict(line=dict(width=1, color="white"), opacity=0.9))
    fig.update_layout(
        height=640, margin=dict(l=10, r=10, t=80, b=10),
        font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=viz.INK),
        coloraxis_colorbar=dict(title="Delay %", ticksuffix="%"),
        geo=dict(landcolor="#f0efec", showlakes=True, lakecolor="#ffffff",
                 subunitcolor=viz.AXIS, countrycolor=viz.AXIS, bgcolor="white"),
    )
    return fig


def export_static_map(fig, path: Path, airport_map: pd.DataFrame) -> str:
    """PNG version of the interactive map for the README (GitHub cannot run Plotly JS).

    Tries Plotly's static export (kaleido); if Chrome/kaleido is unavailable it
    downloads Chrome once, and as a last resort draws a matplotlib fallback.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.write_image(path, width=1300, height=720, scale=2)
        return "plotly"
    except Exception as first_error:
        try:
            import kaleido
            kaleido.get_chrome_sync()
            fig.write_image(path, width=1300, height=720, scale=2)
            return "plotly"
        except Exception:
            print(f"Plotly static export unavailable ({type(first_error).__name__}); "
                  "using matplotlib fallback.")
    _matplotlib_map(airport_map, path)
    return "matplotlib"


STATES_URL = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_20m.zip"


def _draw_state_outlines(ax, cache_path: Path = cfg.EXTERNAL_DIR / "us_states_20m.zip") -> bool:
    """Contiguous-US state borders (US Census, public domain) if geopandas is available."""
    try:
        import geopandas as gpd

        cache_path = Path(cache_path)
        if not cache_path.exists():
            resp = requests.get(STATES_URL, timeout=60)
            resp.raise_for_status()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(resp.content)
        states = gpd.read_file(cache_path)
        states = states[~states["STUSPS"].isin(["AK", "HI", "PR"])]
        states.plot(ax=ax, color="#f0efec", edgecolor=viz.AXIS, linewidth=0.6, zorder=0)
        return True
    except Exception:
        return False


def _matplotlib_map(airport_map: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    conus = airport_map[airport_map["Longitude"].between(-126, -66)
                        & airport_map["Latitude"].between(24, 50)]
    fig, ax = plt.subplots(figsize=(13, 7.2))
    if _draw_state_outlines(ax):
        ax.grid(False)
        ax.set_axis_off()
    sizes = 20 + 900 * conus["Events"] / conus["Events"].max()
    sc = ax.scatter(conus["Longitude"], conus["Latitude"], s=sizes, c=conus["Delay_Pct"],
                    cmap=viz.SEQ_CMAP, edgecolors="white", linewidths=1, alpha=0.9)
    for _, r in conus.nlargest(15, "Events").iterrows():
        ax.annotate(r["Origin"], (r["Longitude"], r["Latitude"]), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color=viz.INK_2)
    fig.colorbar(sc, ax=ax, label="Delay-generating turnarounds (%)", shrink=0.8)
    ax.set_aspect(1 / np.cos(np.radians(38)))
    ax.set_xlim(-126, -66)
    ax.set_ylim(24, 50)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("US airport turnaround risk: contiguous US (bubble size = turnaround volume)",
                 loc="left", fontsize=13, fontweight="bold")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

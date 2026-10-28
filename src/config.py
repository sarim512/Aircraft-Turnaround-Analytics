"""Project-wide paths and constants.

Every path is derived from the repository root, so the project runs the same way
on a laptop, a server or Google Colab -- no hard-coded `/content/...` paths.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths (all relative to the repository root)
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"              # monthly BTS extracts (parquet, git-ignored)
EXTERNAL_DIR = DATA_DIR / "external"    # airport coordinates (git-ignored)
PROCESSED_DIR = DATA_DIR / "processed"  # turnaround dataset cache (git-ignored)

IMAGES_DIR = ROOT_DIR / "images"        # static figures embedded in the README
OUTPUTS_DIR = ROOT_DIR / "outputs"      # result tables + metrics (small, committed)
DOCS_DIR = ROOT_DIR / "docs"            # interactive HTML, servable via GitHub Pages

# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #
YEAR = 2024
MONTHS = tuple(range(1, 13))

BTS_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)
AIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"

# Only the columns the analysis needs -- keeps memory low on a free Colab runtime.
BTS_COLUMNS = [
    "FlightDate",                    # scheduled departure date (local)
    "IATA_CODE_Reporting_Airline",   # carrier code (AA, DL, UA, ...)
    "Tail_Number",                   # aircraft registration -> links consecutive legs
    "Origin",
    "Dest",
    "CRSDepTime",                    # scheduled departure, local HHMM
    "CRSArrTime",                    # scheduled arrival, local HHMM
    "DepDelay",                      # signed departure delay (min)
    "ArrDelay",                      # signed arrival delay (min)
    "CRSElapsedTime",                # scheduled block time (min)
    "Distance",                      # miles
    "Cancelled",
    "Diverted",
]

# --------------------------------------------------------------------------- #
# Turnaround definition
# --------------------------------------------------------------------------- #
MIN_SCHED_TAT = 15      # scheduled ground time window (min). Below = data error,
MAX_SCHED_TAT = 360     # above = overnight / maintenance parking, not a turnaround.
MIN_ACTUAL_TAT = 5      # sanity bounds on realised ground time
MAX_ACTUAL_TAT = 720
DELAY_THRESHOLD = 15    # FAA / US DOT on-time threshold (min)

# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def ensure_dirs() -> None:
    """Create every output directory the pipeline writes to."""
    for d in (RAW_DIR, EXTERNAL_DIR, PROCESSED_DIR, IMAGES_DIR, OUTPUTS_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)

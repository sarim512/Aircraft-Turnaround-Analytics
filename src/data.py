"""Data acquisition, cleaning and turnaround reconstruction.

Pipeline
--------
BTS monthly zip --> parquet extract --> clean flights --> absolute timestamps
--> link consecutive legs of the same aircraft --> turnaround events.

A *turnaround* is the ground time between an aircraft's arrival at an airport
and its next departure from that same airport.
"""
from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from pandas.api.types import union_categoricals

from . import config as cfg

AIRPORT_COLS = ("Origin", "Dest")
CATEGORY_COLS = ("IATA_CODE_Reporting_Airline", "Tail_Number") + AIRPORT_COLS
ONE_MIN = pd.Timedelta(minutes=1)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def hhmm_to_minutes(series: pd.Series) -> pd.Series:
    """Convert BTS local clock times (HHMM, e.g. ``1435``) to minutes after midnight.

    * ``2400`` (BTS's encoding of end-of-day midnight) maps to ``1440`` so that
      ``date + minutes`` correctly lands on the next calendar day.
    * Missing or malformed values (minute part >= 60, hour > 24) become ``NaN``
      instead of being silently mapped to 00:00.
    """
    s = pd.to_numeric(series, errors="coerce").astype("float64")
    hours, mins = s // 100, s % 100
    out = hours * 60 + mins
    bad = (mins >= 60) | (hours > 24) | (out > 1440) | (s < 0)
    return out.mask(bad)


def _values_equal(a: pd.Series, b: pd.Series) -> np.ndarray:
    """Null-safe element-wise equality (missing never equals anything).

    Fast path compares integer codes when both columns are categoricals sharing a
    category set (as produced by ``load_raw``); otherwise compares as strings.
    """
    if (isinstance(a.dtype, pd.CategoricalDtype) and isinstance(b.dtype, pd.CategoricalDtype)
            and a.cat.categories.equals(b.cat.categories)):
        ca, cb = a.cat.codes.to_numpy(), b.cat.codes.to_numpy()
        return (ca == cb) & (ca >= 0)
    sa = a.astype("string").fillna("\x00missing-a").to_numpy(dtype=object)
    sb = b.astype("string").fillna("\x00missing-b").to_numpy(dtype=object)
    return sa == sb


# --------------------------------------------------------------------------- #
# Acquisition
# --------------------------------------------------------------------------- #
def download_month(year: int, month: int, raw_dir: Path = cfg.RAW_DIR,
                   retries: int = 3, timeout: int = 180) -> Path:
    """Download one month of BTS On-Time data and store the needed columns as parquet.

    The CSV is read straight out of the zip archive (no extraction to disk) and only
    ``cfg.BTS_COLUMNS`` are kept, so each month shrinks from ~250 MB of CSV to ~30 MB
    of parquet. Existing files are reused, which makes reruns instant.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"bts_{year}_{month:02d}.parquet"
    if out_path.exists():
        return out_path

    url = cfg.BTS_URL.format(year=year, month=month)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            buffer = io.BytesIO()
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    buffer.write(chunk)
            with zipfile.ZipFile(buffer) as zf:
                member = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                with zf.open(member) as fh:
                    df = pd.read_csv(fh, usecols=lambda c: c in cfg.BTS_COLUMNS,
                                     low_memory=False)
            missing = set(cfg.BTS_COLUMNS) - set(df.columns)
            if missing:
                raise ValueError(f"columns missing from BTS file: {sorted(missing)}")
            df.to_parquet(out_path, index=False)
            return out_path
        except Exception as err:  # network hiccups are common on transtats
            last_error = err
            time.sleep(5 * attempt)
    raise RuntimeError(f"Could not download {year}-{month:02d} from {url}: {last_error}")


def download_bts(year: int = cfg.YEAR, months: Iterable[int] = cfg.MONTHS,
                 raw_dir: Path = cfg.RAW_DIR) -> list[Path]:
    """Download all requested months; fail loudly if any month is missing.

    Printing an error and carrying on would make a failed month silently disappear
    from every downstream statistic.
    """
    paths, failures = [], []
    for m in months:
        try:
            path = download_month(year, m, raw_dir)
            paths.append(path)
            print(f"  {year}-{m:02d}  ok  ({path.stat().st_size / 1e6:.0f} MB)")
        except RuntimeError as err:
            failures.append(m)
            print(f"  {year}-{m:02d}  FAILED  {err}")
    if failures:
        raise RuntimeError(f"Months not downloaded: {failures}. Re-run this cell to retry.")
    return paths


def load_raw(paths: Iterable[Path]) -> pd.DataFrame:
    """Concatenate monthly parquet files with memory-efficient categorical columns."""
    frames = [pd.read_parquet(p) for p in paths]
    if not frames:
        raise FileNotFoundError("No BTS parquet files found -- run the download step first.")

    # Origin and Dest share one category set so they can be compared directly.
    airport_cats = union_categoricals(
        [f[c].astype(str).astype("category") for f in frames for c in AIRPORT_COLS]
    ).categories
    for col in ("IATA_CODE_Reporting_Airline", "Tail_Number"):
        cats = union_categoricals(
            [f[col].astype("string").fillna("").astype("category") for f in frames]
        ).categories
        for f in frames:
            f[col] = pd.Categorical(f[col].astype("string").fillna(""), categories=cats)
    for f in frames:
        for col in AIRPORT_COLS:
            f[col] = pd.Categorical(f[col].astype(str), categories=airport_cats)

    df = pd.concat(frames, ignore_index=True)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    for col in ("DepDelay", "ArrDelay", "CRSElapsedTime", "Distance", "Cancelled", "Diverted"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
    return df


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def _scheduled_arrival(df: pd.DataFrame) -> pd.Series:
    """Scheduled arrival as an absolute (destination-local) timestamp.

    ``CRSArrTime`` is a clock time with no date. The arrival date is recovered by
    comparing it with ``departure + scheduled block time``: the two differ only by
    the time-zone offset (< 12 h), so rounding the gap to whole days gives the
    correct date shift -- including red-eyes and short westbound hops that
    'arrive before they leave' on the clock.
    """
    candidate = df["FlightDate"] + pd.to_timedelta(df["CRSArr_Min"], unit="min")
    expected = df["Sched_Dep"] + pd.to_timedelta(df["CRSElapsedTime"], unit="min")
    shift = np.round((expected - candidate) / pd.Timedelta(days=1))
    fallback = pd.Series(
        np.where(df["CRSArr_Min"] < df["CRSDep_Min"] - 120, 1.0, 0.0), index=df.index
    )
    shift = shift.fillna(fallback)
    return candidate + pd.to_timedelta(shift, unit="D")


def clean_flights(df: pd.DataFrame) -> tuple[pd.DataFrame, list[tuple[str, int]]]:
    """Validate flights, derive schedule features and absolute timestamps.

    Returns the cleaned frame and a funnel log of row counts per step.
    """
    log: list[tuple[str, int]] = [("Raw flight records", len(df))]
    df = df.copy()

    df["CRSDep_Min"] = hhmm_to_minutes(df["CRSDepTime"])
    df["CRSArr_Min"] = hhmm_to_minutes(df["CRSArrTime"])
    df = df.dropna(subset=["FlightDate", "CRSDep_Min", "CRSArr_Min", "Origin", "Dest"])
    df["Sched_Dep"] = df["FlightDate"] + pd.to_timedelta(df["CRSDep_Min"], unit="min")

    # Schedule-only features are computed on the FULL published schedule, before
    # cancellations are removed. Counting only flights that actually operated would
    # leak day-of disruption (storm days have fewer flights) into the features.
    df["Origin_Daily_Deps"] = (
        df.groupby(["Origin", "FlightDate"], observed=True)["Sched_Dep"]
        .transform("size").astype("int32")
    )

    tails = df["Tail_Number"].astype("category")
    good_tails = [t for t in tails.cat.categories
                  if isinstance(t, str) and len(t.strip()) > 2 and t.strip() != "0"]
    df = df[tails.isin(good_tails)].copy()
    df["Tail_Number"] = df["Tail_Number"].astype("category").cat.remove_unused_categories()
    log.append(("Valid tail number", len(df)))

    df = df.sort_values(["Tail_Number", "Sched_Dep"], kind="mergesort")
    df["Leg_Of_Day"] = (
        df.groupby(["Tail_Number", "FlightDate"], observed=True).cumcount() + 1
    ).astype("int16")

    df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)]
    log.append(("Operated (not cancelled / diverted)", len(df)))

    df = df.dropna(subset=["DepDelay", "ArrDelay"])
    log.append(("Complete delay information", len(df)))

    df["Sched_Arr"] = _scheduled_arrival(df)
    df["Actual_Dep"] = df["Sched_Dep"] + pd.to_timedelta(df["DepDelay"], unit="min")
    df["Actual_Arr"] = df["Sched_Arr"] + pd.to_timedelta(df["ArrDelay"], unit="min")
    df["HourDep"] = (df["CRSDep_Min"] // 60).clip(upper=23).astype("int8")
    df["Month"] = df["FlightDate"].dt.month.astype("int8")
    df["DayOfWeek"] = df["FlightDate"].dt.dayofweek.astype("int8")  # 0 = Mon
    return df.reset_index(drop=True), log


# --------------------------------------------------------------------------- #
# Turnaround reconstruction
# --------------------------------------------------------------------------- #
def build_turnarounds(df: pd.DataFrame, return_diagnostics: bool = False):
    """Link each departure to the same aircraft's previous arrival at that airport.

    Pitfalls avoided:

    * Ground time is computed from **absolute timestamps** (date + clock + signed
      delay), not clock-minute differences modulo 1440. The modulo version made an
      aircraft that landed Monday 10:00 and left Wednesday 11:00 look like a 60-min
      turnaround.
    * Events are selected on the **scheduled** ground time (known in advance).
      Filtering on the *actual* ground time selects on the outcome, which silently
      drops the worst delays and biases every downstream statistic.
    * The delay label requires the outbound departure itself to be late (see
      ``TAT_Delayed`` below). The naive label ``actual > scheduled`` flags every
      aircraft that landed early and simply waited for its slot.
    """
    df = df.sort_values(["Tail_Number", "Sched_Dep"], kind="mergesort").reset_index(drop=True)
    prev = df[["Tail_Number", "Dest", "Origin", "Sched_Arr", "Actual_Arr", "ArrDelay",
               "CRSArr_Min"]].shift(1)

    linked = _values_equal(df["Tail_Number"], prev["Tail_Number"]) & \
        _values_equal(df["Origin"], prev["Dest"])

    tat_sched = (df["Sched_Dep"] - prev["Sched_Arr"]) / ONE_MIN
    tat_actual = (df["Actual_Dep"] - prev["Actual_Arr"]) / ONE_MIN
    keep = (
        linked
        & tat_sched.between(cfg.MIN_SCHED_TAT, cfg.MAX_SCHED_TAT).to_numpy()
        & tat_actual.between(cfg.MIN_ACTUAL_TAT, cfg.MAX_ACTUAL_TAT).to_numpy()
    )

    cols = ["Tail_Number", "FlightDate", "IATA_CODE_Reporting_Airline", "Origin", "Dest",
            "Sched_Dep", "Month", "DayOfWeek", "HourDep", "Distance", "Leg_Of_Day",
            "Origin_Daily_Deps", "DepDelay", "ArrDelay"]
    out = df.loc[keep, cols].rename(columns={"IATA_CODE_Reporting_Airline": "Airline"})
    out["Prev_Origin"] = prev.loc[keep, "Origin"].astype(str).astype("category")
    out["Prev_ArrDelay"] = prev.loc[keep, "ArrDelay"].astype("float32")
    out["TAT_Scheduled"] = tat_sched[keep].astype("float32")
    out["TAT_Actual"] = tat_actual[keep].astype("float32")
    # Identity: TAT_Delay == DepDelay - Prev_ArrDelay (ground-time overrun vs plan).
    out["TAT_Delay"] = (out["TAT_Actual"] - out["TAT_Scheduled"]).astype("float32")
    # Delay-generating turnaround: the ground phase overran its plan by >= 15 min AND
    # the outbound flight left >= 15 min late. Early arrivals that wait for their
    # slot are no longer counted; propagated delay that the crew absorbed is not
    # counted either.
    out["TAT_Delayed"] = (
        (out["TAT_Delay"] >= cfg.DELAY_THRESHOLD) & (out["DepDelay"] >= cfg.DELAY_THRESHOLD)
    ).astype("int8")
    for c in ("Airline", "Origin", "Dest", "Tail_Number"):
        out[c] = out[c].astype(str).astype("category")
    out = out.reset_index(drop=True)

    if not return_diagnostics:
        return out
    return out, _naive_diagnostics(df, prev, linked, tat_actual, keep, out)


def _naive_diagnostics(df, prev, linked, tat_actual, keep, out) -> dict:
    """Re-create the naive clock-modulo method to quantify what it gets wrong."""
    def tod(ts):  # minutes after local midnight
        return (ts - ts.dt.normalize()) / ONE_MIN

    naive_act = (tod(df["Actual_Dep"]) - tod(prev["Actual_Arr"])) % 1440
    naive_sched = (df["CRSDep_Min"] - prev["CRSArr_Min"]) % 1440
    naive_keep = linked & naive_act.between(15, 360).to_numpy()
    aliased = naive_keep & (tat_actual > 24 * 60).to_numpy()
    naive_label = (naive_act - naive_sched)[naive_keep] > 0
    return {
        "naive_events": int(naive_keep.sum()),
        "naive_multi_day_aliased": int(aliased.sum()),
        "naive_delayed_rate": float(naive_label.mean()),
        "events": int(keep.sum()),
        "delayed_rate": float(out["TAT_Delayed"].mean()),
        "share_ground_time_over_plan": float((out["TAT_Delay"] > 0).mean()),
    }


# --------------------------------------------------------------------------- #
# Orchestration with caching
# --------------------------------------------------------------------------- #
def load_or_build_turnarounds(year: int = cfg.YEAR, months: Iterable[int] = cfg.MONTHS,
                              force_rebuild: bool = False):
    """Return ``(turnarounds, funnel_log, diagnostics)``, building and caching on first run."""
    cfg.ensure_dirs()
    cache = cfg.PROCESSED_DIR / f"turnarounds_{year}.parquet"
    meta = cfg.PROCESSED_DIR / f"turnarounds_{year}_meta.json"
    if cache.exists() and meta.exists() and not force_rebuild:
        info = json.loads(meta.read_text())
        print(f"Loaded cached turnaround dataset: {cache.relative_to(cfg.ROOT_DIR)}")
        return pd.read_parquet(cache), info["funnel"], info["diagnostics"]

    print(f"Downloading BTS On-Time Performance data for {year}...")
    paths = download_bts(year, months)
    raw = load_raw(paths)
    flights, funnel = clean_flights(raw)
    del raw
    tat, diagnostics = build_turnarounds(flights, return_diagnostics=True)
    del flights
    funnel.append(("Valid turnaround events", len(tat)))

    tat.to_parquet(cache, index=False)
    meta.write_text(json.dumps({"funnel": funnel, "diagnostics": diagnostics}, indent=2))
    return tat, funnel, diagnostics

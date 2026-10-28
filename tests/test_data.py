"""Unit tests for the turnaround reconstruction logic.

Each test pins down one failure mode of a naive clock-modulo approach.
Run with:  pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from src import analysis
from src.data import build_turnarounds, clean_flights, hhmm_to_minutes


def flight(tail, date, origin, dest, dep, arr, elapsed, dep_delay=0.0, arr_delay=0.0,
           cancelled=0.0, diverted=0.0, airline="XX"):
    return {"FlightDate": pd.Timestamp(date), "IATA_CODE_Reporting_Airline": airline,
            "Tail_Number": tail, "Origin": origin, "Dest": dest, "CRSDepTime": dep,
            "CRSArrTime": arr, "DepDelay": dep_delay, "ArrDelay": arr_delay,
            "CRSElapsedTime": elapsed, "Distance": 500.0, "Cancelled": cancelled,
            "Diverted": diverted}


def turnarounds(rows):
    flights, _ = clean_flights(pd.DataFrame(rows))
    return build_turnarounds(flights)


# --------------------------------------------------------------------------- #
def test_hhmm_to_minutes():
    out = hhmm_to_minutes(pd.Series([0, 5, 930, 1259, 2400, np.nan, 1275, -5]))
    assert out.iloc[:5].tolist() == [0, 5, 570, 779, 1440]
    assert out.iloc[5:].isna().all()          # missing / invalid are NaN, not midnight


def test_simple_turnaround():
    t = turnarounds([
        flight("N101", "2024-03-01", "AAA", "BBB", 800, 1000, 120),
        flight("N101", "2024-03-01", "BBB", "CCC", 1045, 1200, 75),
    ])
    assert len(t) == 1
    assert t.loc[0, "TAT_Scheduled"] == 45 and t.loc[0, "TAT_Actual"] == 45
    assert t.loc[0, "Leg_Of_Day"] == 2


def test_turnaround_across_midnight():
    # arrives 23:40, inbound 30 min late (00:10 next day); departs 00:40 sched, 35 min late
    t = turnarounds([
        flight("N101", "2024-03-01", "AAA", "BBB", 2140, 2340, 120, arr_delay=30),
        flight("N101", "2024-03-02", "BBB", "CCC", 40, 200, 80, dep_delay=35),
    ])
    assert len(t) == 1
    assert t.loc[0, "TAT_Scheduled"] == 60
    assert t.loc[0, "TAT_Actual"] == 65


def test_multi_day_gap_is_not_a_turnaround():
    # lands Monday 10:00, next leaves Wednesday 11:00: modulo-1440 would call this 60 min
    t = turnarounds([
        flight("N101", "2024-03-04", "AAA", "BBB", 800, 1000, 120),
        flight("N101", "2024-03-06", "BBB", "CCC", 1100, 1300, 120),
    ])
    assert len(t) == 0


def test_red_eye_arrival_date():
    # departs 22:00, arrives 01:00 next day (CRSArr < CRSDep): turnaround next morning
    t = turnarounds([
        flight("N101", "2024-03-01", "AAA", "BBB", 2200, 100, 180),
        flight("N101", "2024-03-02", "BBB", "CCC", 200, 400, 120),
    ])
    assert len(t) == 1 and t.loc[0, "TAT_Scheduled"] == 60


def test_westbound_hop_arriving_before_departure_clock():
    # 08:00 departure, 07:55 local arrival one time zone west: same day, not next day
    t = turnarounds([
        flight("N101", "2024-03-01", "AAA", "BBB", 800, 755, 55),
        flight("N101", "2024-03-01", "BBB", "CCC", 840, 1000, 80),
    ])
    assert len(t) == 1 and t.loc[0, "TAT_Scheduled"] == 45


def test_early_arrival_waiting_is_not_a_delay():
    # lands 20 min early, departs on time: ground time 20 min over plan, but no delay generated
    t = turnarounds([
        flight("N101", "2024-03-01", "AAA", "BBB", 800, 1000, 120, arr_delay=-20),
        flight("N101", "2024-03-01", "BBB", "CCC", 1045, 1200, 75, dep_delay=0),
    ])
    assert t.loc[0, "TAT_Delay"] == 20
    assert t.loc[0, "TAT_Delayed"] == 0


def test_ground_generated_delay_is_flagged():
    t = turnarounds([
        flight("N101", "2024-03-01", "AAA", "BBB", 800, 1000, 120, arr_delay=0),
        flight("N101", "2024-03-01", "BBB", "CCC", 1045, 1200, 75, dep_delay=25),
    ])
    assert t.loc[0, "TAT_Delayed"] == 1


def test_absorbed_inbound_delay_is_not_flagged():
    # inbound 40 min late, crew turns fast, departs 30 min late: ground phase recovered time
    t = turnarounds([
        flight("N101", "2024-03-01", "AAA", "BBB", 800, 1000, 120, arr_delay=40),
        flight("N101", "2024-03-01", "BBB", "CCC", 1100, 1200, 60, dep_delay=30),
    ])
    assert t.loc[0, "TAT_Delayed"] == 0


def test_airport_mismatch_and_cancellations_are_excluded():
    t = turnarounds([
        flight("N101", "2024-03-01", "AAA", "BBB", 800, 1000, 120),
        flight("N101", "2024-03-01", "ZZZ", "CCC", 1045, 1200, 75),   # tail-swap / data gap
        flight("N202", "2024-03-01", "AAA", "BBB", 800, 1000, 120, cancelled=1),
        flight("N202", "2024-03-01", "BBB", "CCC", 1045, 1200, 75),
    ])
    assert len(t) == 0


def test_daily_volume_counts_cancelled_flights():
    flights, _ = clean_flights(pd.DataFrame([
        flight("N101", "2024-03-01", "AAA", "BBB", 800, 1000, 120),
        flight("N202", "2024-03-01", "AAA", "BBB", 900, 1100, 120, cancelled=1),
    ]))
    assert (flights["Origin_Daily_Deps"] == 2).all()


# --------------------------------------------------------------------------- #
def test_jackknife_matches_textbook_se_for_singleton_clusters():
    rng = np.random.default_rng(0)
    x = rng.lognormal(4, 0.5, size=2_000)
    se = analysis.cluster_jackknife_se(x, np.arange(len(x)))
    assert se == pytest.approx(x.std(ddof=1) / np.sqrt(len(x)), rel=1e-9)


def test_cluster_bootstrap_widens_with_correlated_clusters():
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(100), 200)
    x = rng.normal(0, 1, 100)[clusters] + rng.normal(0, 0.2, clusters.size)
    boot = analysis.cluster_bootstrap_mean(x, clusters, n_boot=500)
    naive = x.std(ddof=1) / np.sqrt(x.size)
    assert boot["boot_se"] > 5 * naive


def test_entropy_uses_common_bins():
    edges = analysis.tat_bin_edges(5)
    narrow = np.full(1000, 42.0)
    assert analysis.shannon_entropy(narrow, edges) == pytest.approx(0.0)
    uniform = np.arange(0, 720, 5) + 2.5
    assert analysis.shannon_entropy(uniform, edges) == pytest.approx(np.log2(len(edges) - 1))

"""Predictive modeling of delay-generating turnarounds.

Three nested feature tiers measure the *value of information* as the day of
operation approaches:

=========  ==========================================  ==============================
Tier       Available when                              Adds
=========  ==========================================  ==============================
Calendar   weeks ahead (network planning)              airline, airport, month, weekday, hour
Schedule   day of operation (ops control)              planned ground time, distance,
                                                       leg of day, airport daily volume
Real-time  inbound aircraft is on approach / landed    inbound arrival delay
=========  ==========================================  ==============================
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from . import config as cfg

TARGET = "TAT_Delayed"
CATEGORICAL = ["Airline", "Origin"]
_CALENDAR = ["Airline", "Origin", "Month", "DayOfWeek", "HourDep"]
_SCHEDULE = _CALENDAR + ["TAT_Scheduled", "Distance", "Leg_Of_Day", "Origin_Daily_Deps"]
_REALTIME = _SCHEDULE + ["Prev_ArrDelay"]
FEATURE_TIERS = {"Calendar": _CALENDAR, "Schedule": _SCHEDULE, "Real-time": _REALTIME}
TIER_LABELS = {
    "Calendar": "Weeks ahead\n(calendar only)",
    "Schedule": "Day of operation\n(full schedule)",
    "Real-time": "Inbound landed\n(real-time)",
}

FEATURE_NAMES = {
    "Prev_ArrDelay": "Inbound arrival delay",
    "TAT_Scheduled": "Planned ground time",
    "HourDep": "Departure hour",
    "Leg_Of_Day": "Leg of the day",
    "Origin_Daily_Deps": "Airport daily departures",
    "DayOfWeek": "Day of week",
    "Month": "Month",
    "Distance": "Outbound distance",
    "Airline": "Airline",
    "Origin": "Airport",
}


def make_model_frame(df: pd.DataFrame, top_airports: int = 200):
    """Feature matrix, target and grouping key (flight date) for the split.

    Airports outside the ``top_airports`` busiest are pooled as ``OTHER`` so the
    categorical stays below the gradient-boosting bin limit (255).
    """
    X = df[_REALTIME].copy()
    busiest = df["Origin"].value_counts().head(top_airports).index
    origin = df["Origin"].astype(str)
    X["Origin"] = pd.Categorical(origin.where(origin.isin(busiest), "OTHER"))
    X["Airline"] = pd.Categorical(df["Airline"].astype(str))
    X["Prev_ArrDelay"] = X["Prev_ArrDelay"].clip(-60, 300)
    for c in ("Month", "DayOfWeek", "HourDep", "Leg_Of_Day", "Origin_Daily_Deps"):
        X[c] = X[c].astype("int16")
    y = df[TARGET].to_numpy(dtype="int8")
    groups = df["FlightDate"].to_numpy()
    return X, y, groups


def date_grouped_split(groups, test_size: float = 0.2, random_state: int = cfg.RANDOM_STATE):
    """Hold out whole calendar days.

    A row-level random split puts flights from the same storm day, the same aircraft
    rotation and the same airport bank into both train and test, so the score
    partly measures memorisation of that day. Holding out complete days keeps
    every month represented while removing that leak.
    """
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(np.zeros(len(groups)), groups=groups))
    return train_idx, test_idx


def train_tier(X_train: pd.DataFrame, y_train, tier: str, max_rows: int | None = 2_000_000,
               random_state: int = cfg.RANDOM_STATE) -> HistGradientBoostingClassifier:
    """Gradient-boosted trees on one feature tier.

    ``HistGradientBoostingClassifier`` rather than a random forest: it trains in minutes on millions of rows on a 2-core Colab CPU, handles the
    airline / airport categoricals natively, and uses early stopping.
    """
    features = FEATURE_TIERS[tier]
    if max_rows and len(X_train) > max_rows:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_train), size=max_rows, replace=False)
        X_train, y_train = X_train.iloc[idx], np.asarray(y_train)[idx]
    model = HistGradientBoostingClassifier(
        learning_rate=0.1, max_iter=400, max_leaf_nodes=63, min_samples_leaf=200,
        l2_regularization=1.0, categorical_features="from_dtype",
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=random_state,
    )
    model.fit(X_train[features], y_train)
    return model


def evaluate(y_true, prob) -> dict:
    """Ranking (ROC-AUC), minority-class precision (PR-AUC) and calibration (Brier)."""
    y_true = np.asarray(y_true)
    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "brier": brier_score_loss(y_true, prob),
        "base_rate": float(y_true.mean()),
    }


def alert_threshold(prob_reference, rate: float) -> float:
    """Threshold that flags the riskiest ``rate`` share of turnarounds.

    Set from *training* predictions with the training base rate, so the test set
    is never used to tune it (a fixed cut-off such as 0.5 ignores class imbalance).
    """
    return float(np.quantile(prob_reference, 1 - rate))


def permutation_importance_table(model, X: pd.DataFrame, y, tier: str, n_rows: int = 100_000,
                                 n_repeats: int = 3,
                                 random_state: int = cfg.RANDOM_STATE) -> pd.DataFrame:
    """Drop in ROC-AUC when each feature is shuffled, on held-out data.

    Impurity-based importances (``feature_importances_``) are computed
    on training data and are biased toward continuous, high-cardinality features.
    """
    features = FEATURE_TIERS[tier]
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(n_rows, len(X)), replace=False)
    result = permutation_importance(model, X.iloc[idx][features], np.asarray(y)[idx],
                                    scoring="roc_auc", n_repeats=n_repeats,
                                    random_state=random_state, n_jobs=1)
    return (pd.DataFrame({"Feature": [FEATURE_NAMES.get(f, f) for f in features],
                          "Importance": result.importances_mean,
                          "Std": result.importances_std})
            .sort_values("Importance", ascending=False).reset_index(drop=True))


def auc_by_group(frame: pd.DataFrame, group: str, y_col: str, prob_cols: list[str],
                 min_n: int = 500) -> pd.DataFrame:
    """ROC-AUC of several probability columns within each group."""
    rows = []
    for g, sub in frame.groupby(group, observed=True):
        if len(sub) < min_n or sub[y_col].nunique() < 2:
            continue
        row = {group: g, "N": len(sub), "Delay_Rate": sub[y_col].mean()}
        row.update({c: roc_auc_score(sub[y_col], sub[c]) for c in prob_cols})
        rows.append(row)
    return pd.DataFrame(rows)

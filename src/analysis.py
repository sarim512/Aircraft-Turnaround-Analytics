"""Statistical analysis: distribution fitting, information theory, uncertainty
quantification and unsupervised segmentation."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import config as cfg

# --------------------------------------------------------------------------- #
# Distribution modeling
# --------------------------------------------------------------------------- #
PARAMETRIC = {
    "Normal": stats.norm,
    "Exponential": stats.expon,
    "Gamma": stats.gamma,
    "Weibull": stats.weibull_min,
    "Log-normal": stats.lognorm,
}


def fit_distributions(x, sample_size: int = 300_000, mixture_components=(2, 3),
                      random_state: int = cfg.RANDOM_STATE):
    """Fit candidate distributions by maximum likelihood and rank them by AIC / BIC.

    Notes
    -----
    * Fitting runs on a random subsample: with millions of points every
      goodness-of-fit *p*-value is ~0 (any tiny misfit is "significant"), so the KS
      **statistic** is reported as an effect size instead. ``ΔAIC`` relative to the
      best model is what should be compared -- raw AIC values are ~10^7 and look
      identical on a bar chart.
    * Log-normal mixtures (Gaussian mixtures on log TAT) are included because
      turnaround times are visibly multimodal (regional quick-turns vs. mainline).

    Returns ``(results_df, pdfs)`` where ``pdfs[name]`` is a callable density.
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if len(x) > sample_size:
        x = rng.choice(x, size=sample_size, replace=False)
    n = len(x)

    rows, pdfs = [], {}
    for name, dist in PARAMETRIC.items():
        params = dist.fit(x)
        loglik = float(np.sum(dist.logpdf(x, *params)))
        ks = stats.kstest(x, dist.cdf, args=params).statistic
        rows.append(_score(name, len(params), loglik, n, ks))
        pdfs[name] = _frozen_pdf(dist, params)

    logx = np.log(x).reshape(-1, 1)
    for m in mixture_components:
        gm = GaussianMixture(n_components=m, n_init=2, random_state=random_state).fit(logx)
        # change of variables: f_X(x) = f_logX(log x) / x
        loglik = float(np.sum(gm.score_samples(logx) - logx.ravel()))
        w, mu, sd = gm.weights_, gm.means_.ravel(), np.sqrt(gm.covariances_.ravel())

        def cdf(t, w=w, mu=mu, sd=sd):
            t = np.log(np.asarray(t, dtype=float))[..., None]
            return np.sum(w * stats.norm.cdf((t - mu) / sd), axis=-1)

        ks = stats.kstest(x, cdf).statistic
        name = f"Log-normal mixture (k={m})"
        rows.append(_score(name, 3 * m - 1, loglik, n, ks))
        pdfs[name] = _mixture_pdf(gm)

    res = pd.DataFrame(rows).sort_values("AIC").reset_index(drop=True)
    res["dAIC"] = res["AIC"] - res["AIC"].min()
    return res, pdfs


def _score(name, k, loglik, n, ks):
    return {"Distribution": name, "Params": k, "LogLik": loglik,
            "AIC": 2 * k - 2 * loglik, "BIC": k * np.log(n) - 2 * loglik, "KS_stat": ks}


def _frozen_pdf(dist, params) -> Callable:
    return lambda t: dist.pdf(t, *params)


def _mixture_pdf(gm) -> Callable:
    def pdf(t):
        t = np.asarray(t, dtype=float)
        return np.exp(gm.score_samples(np.log(t).reshape(-1, 1))) / t
    return pdf


# --------------------------------------------------------------------------- #
# Information theory
# --------------------------------------------------------------------------- #
def tat_bin_edges(width: int = 5) -> np.ndarray:
    """Common bin edges for every group, so entropies are comparable.

    Binning each group over its own range would give a group with a wider range
    wider bins, which changes the entropy estimate by itself.
    """
    return np.arange(0, cfg.MAX_ACTUAL_TAT + width, width, dtype=float)


def _histogram_pmf(values, edges, eps: float = 0.0) -> np.ndarray:
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=edges)
    p = counts.astype(float) + eps
    return p / p.sum()


def shannon_entropy(values, edges) -> float:
    """Shannon entropy (bits) of the binned distribution."""
    p = _histogram_pmf(values, edges)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def kl_divergence(p_values, q_values, edges, eps: float = 1e-9) -> float:
    """D_KL(P || Q) in bits on shared bins (eps-smoothed so empty bins don't explode)."""
    p = _histogram_pmf(p_values, edges, eps)
    q = _histogram_pmf(q_values, edges, eps)
    return float(np.sum(p * np.log2(p / q)))


def js_distance(p_values, q_values, edges) -> float:
    """Jensen-Shannon distance (bits^0.5): symmetric and bounded in [0, 1]."""
    return float(jensenshannon(_histogram_pmf(p_values, edges),
                               _histogram_pmf(q_values, edges), base=2))


def entropy_by_group(df: pd.DataFrame, group: str, value: str = "TAT_Actual",
                     min_n: int = 1_000, width: int = 5) -> pd.DataFrame:
    edges = tat_bin_edges(width)
    rows = [{group: g, "N": len(s), "Entropy_bits": shannon_entropy(s, edges)}
            for g, s in df.groupby(group, observed=True)[value] if len(s) >= min_n]
    return pd.DataFrame(rows).sort_values("Entropy_bits", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Uncertainty quantification (cluster-aware)
# --------------------------------------------------------------------------- #
def _cluster_sums(values, clusters):
    values = np.asarray(values, dtype=float)
    codes, _ = pd.factorize(np.asarray(clusters))
    sums = np.bincount(codes, weights=values)
    counts = np.bincount(codes).astype(float)
    return sums, counts


def cluster_bootstrap_mean(values, clusters, n_boot: int = 2_000, alpha: float = 0.05,
                           random_state: int = cfg.RANDOM_STATE, chunk: int = 250):
    """Bootstrap CI for a mean, resampling whole clusters (aircraft) with replacement.

    Turnarounds of the same aircraft are correlated (same fleet type, same rotation,
    same crew base). Resampling individual rows treats 1M correlated rows as 1M
    independent ones and understates uncertainty.
    """
    rng = np.random.default_rng(random_state)
    sums, counts = _cluster_sums(values, clusters)
    g = len(sums)
    boots = []
    for start in range(0, n_boot, chunk):
        size = min(chunk, n_boot - start)
        idx = rng.integers(0, g, size=(size, g))
        boots.append(sums[idx].sum(axis=1) / counts[idx].sum(axis=1))
    boots = np.concatenate(boots)
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean": sums.sum() / counts.sum(), "ci_low": lo, "ci_high": hi,
            "boot_se": boots.std(ddof=1), "n_clusters": g}


def cluster_jackknife_se(values, clusters) -> float:
    """Delete-one-cluster jackknife standard error of the mean (exact, vectorised).

    With one observation per cluster this reduces to the textbook s / sqrt(n).
    A common shortcut, leaving out only the first few hundred observations and
    applying the n-observation formula with that small n, understates the
    standard error by more than an order of magnitude on samples of ~10^6.
    """
    sums, counts = _cluster_sums(values, clusters)
    g = len(sums)
    loo = (sums.sum() - sums) / (counts.sum() - counts)
    return float(np.sqrt((g - 1) / g * np.sum((loo - loo.mean()) ** 2)))


def uncertainty_table(df: pd.DataFrame, group: str = "Airline", value: str = "TAT_Actual",
                      cluster: str = "Tail_Number", top_n: int = 10, n_boot: int = 2_000,
                      random_state: int = cfg.RANDOM_STATE) -> pd.DataFrame:
    """Naive (i.i.d.) vs cluster-robust uncertainty for the mean of ``value`` per group."""
    top = df[group].value_counts().head(top_n).index
    rows = []
    for g in top:
        sub = df.loc[df[group] == g, [value, cluster]].dropna()
        v = sub[value].to_numpy(dtype=float)
        boot = cluster_bootstrap_mean(v, sub[cluster].to_numpy(), n_boot=n_boot,
                                      random_state=random_state)
        naive_se = v.std(ddof=1) / np.sqrt(len(v))
        jk_se = cluster_jackknife_se(v, sub[cluster].to_numpy())
        rows.append({group: g, "N": len(v), "Aircraft": boot["n_clusters"],
                     "Mean": boot["mean"], "CI_low": boot["ci_low"], "CI_high": boot["ci_high"],
                     "SE_naive": naive_se, "SE_cluster_boot": boot["boot_se"],
                     "SE_cluster_jackknife": jk_se,
                     "Design_effect": (jk_se / naive_se) ** 2})
    return pd.DataFrame(rows).sort_values("Mean").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Airport segmentation
# --------------------------------------------------------------------------- #
def airport_features(df: pd.DataFrame, min_events: int = 200) -> pd.DataFrame:
    """Per-airport operating profile. Inbound delay uses ``Prev_ArrDelay`` (the
    aircraft arriving *at* this airport), not the outbound flight's arrival delay
    at its *destination*, which describes a different airport."""
    feats = df.groupby("Origin", observed=True).agg(
        Mean_TAT=("TAT_Actual", "mean"),
        Std_TAT=("TAT_Actual", "std"),
        Median_Sched_TAT=("TAT_Scheduled", "median"),
        Delay_Rate=("TAT_Delayed", "mean"),
        Mean_Overrun=("TAT_Delay", "mean"),
        Mean_Inbound_Delay=("Prev_ArrDelay", "mean"),
        Mean_Outbound_Delay=("DepDelay", "mean"),
        Events=("TAT_Actual", "size"),
    ).dropna()
    feats = feats[feats["Events"] >= min_events].copy()
    feats["Log_Events"] = np.log10(feats["Events"])  # volume is heavily right-skewed
    return feats


def segment_airports(feats: pd.DataFrame, n_components: int = 3, k_range=range(2, 9),
                     k: int | None = None, random_state: int = cfg.RANDOM_STATE) -> dict:
    """Standardise -> PCA -> K-Means, choosing k by silhouette unless ``k`` is given."""
    cols = [c for c in feats.columns if c != "Events"]
    X = StandardScaler().fit_transform(feats[cols])
    pca = PCA(n_components=n_components, random_state=random_state)
    Z = pca.fit_transform(X)

    inertias, silhouettes = {}, {}
    for kk in k_range:
        km = KMeans(n_clusters=kk, n_init=10, random_state=random_state).fit(Z)
        inertias[kk] = km.inertia_
        silhouettes[kk] = silhouette_score(Z, km.labels_)
    best_k = k or max(silhouettes, key=silhouettes.get)
    labels = KMeans(n_clusters=best_k, n_init=10, random_state=random_state).fit_predict(Z)

    loadings = pd.DataFrame(pca.components_.T, index=cols,
                            columns=[f"PC{i + 1}" for i in range(n_components)])
    return {"Z": Z, "labels": labels, "best_k": best_k, "inertias": inertias,
            "silhouettes": silhouettes, "explained": pca.explained_variance_ratio_,
            "loadings": loadings, "feature_cols": cols}

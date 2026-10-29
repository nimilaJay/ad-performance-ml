"""Feature engineering, benchmark definition, target construction and chronological splitting.

Each modelling row describes a campaign at the end of day t, using only information available on
or before day t, and the target is whether day t+1's CTR exceeds the benchmark.
"""

import numpy as np
import pandas as pd

from src import config


def _group_key(df: pd.DataFrame) -> pd.Series:
    return df[config.BENCHMARK_GROUP].astype(str).agg(" | ".join, axis=1)


def split_dates(panel: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Pick train/validation cut-off dates so that ~70/15/15% of labelled rows fall in each split."""
    p = panel.sort_values(["campaign_id", "date"])
    is_first_day = p["campaign_id"].ne(p["campaign_id"].shift())
    target_dates = p.loc[~is_first_day & (p["impressions"] >= config.MIN_IMPRESSIONS), "date"].sort_values()
    n = len(target_dates)
    train_end = target_dates.iloc[int(n * config.TRAIN_FRAC) - 1]
    val_end = target_dates.iloc[int(n * (config.TRAIN_FRAC + config.VAL_FRAC)) - 1]
    return train_end, val_end


def compute_benchmark(panel: pd.DataFrame, train_end: pd.Timestamp) -> dict[str, float]:
    """Benchmark CTR (%) per group, using training-period data only so no future data leaks in."""
    if config.BENCHMARK_MODE != "train_median":
        return dict(config.BENCHMARK_MODE)
    hist = panel[(panel["date"] <= train_end) & (panel["impressions"] >= config.MIN_IMPRESSIONS)]
    ctr = 100 * hist["clicks"] / hist["impressions"]
    return ctr.groupby(_group_key(hist)).median().round(4).to_dict()


def build_features(panel: pd.DataFrame, benchmark: dict[str, float],
                   include_unlabelled: bool = False) -> pd.DataFrame:
    """Turn a daily campaign panel (output of data.clean) into modelling rows.

    With include_unlabelled=True, rows whose next day is unknown (e.g. the most recent day of a
    live campaign) are kept with a missing target, so they can be scored.
    """
    df = panel.sort_values(["campaign_id", "date"]).reset_index(drop=True).copy()
    key = _group_key(df)
    unknown = sorted(set(key) - set(benchmark))
    if unknown:
        raise ValueError(f"No benchmark for {unknown}; known groups: {sorted(benchmark)}")
    df["benchmark_ctr"] = key.map(benchmark)
    bench = df["benchmark_ctr"] / 100

    g = df.groupby("campaign_id", sort=False)

    def rolling(col, window, how="sum"):
        r = g[col].rolling(window, min_periods=1)
        return getattr(r, how)().reset_index(level=0, drop=True)

    imp_3, clk_3 = rolling("impressions", 3), rolling("clicks", 3)
    imp_7, clk_7, spend_7 = rolling("impressions", 7), rolling("clicks", 7), rolling("spend", 7)

    with np.errstate(divide="ignore", invalid="ignore"):
        ctr_1d = (df["clicks"] / df["impressions"]).where(df["impressions"] > 0)
        df["ctr_1d"] = 100 * ctr_1d
        df["ctr_7d"] = 100 * (clk_7 / imp_7).where(imp_7 > 0)
        df["ctr_index_1d"] = ctr_1d / bench
        df["ctr_index_3d"] = (clk_3 / imp_3).where(imp_3 > 0) / bench
        df["ctr_index_7d"] = (clk_7 / imp_7).where(imp_7 > 0) / bench
        df["cpm_7d"] = (1000 * spend_7 / imp_7).where(imp_7 > 0)
    # On a day without delivery the 1-day CTR is undefined; fall back to the 7-day index (no news).
    df["ctr_index_1d"] = df["ctr_index_1d"].fillna(df["ctr_index_7d"])
    df["ctr_trend"] = df["ctr_index_1d"] - df["ctr_index_7d"]

    df["campaign_day"] = (df["date"] - g["date"].transform("min")).dt.days + 1
    df["spend_1d"] = df["spend"]
    df["spend_7d_avg"] = rolling("spend", 7, "mean")
    df["spend_cum"] = g["spend"].cumsum()
    df["impressions_1d"] = df["impressions"]
    df["impressions_7d_avg"] = rolling("impressions", 7, "mean")
    df["clicks_1d"] = df["clicks"]
    df["clicks_7d_avg"] = rolling("clicks", 7, "mean")

    # Target: next calendar day's CTR vs. benchmark (the panel has one row per calendar day).
    df["target_date"] = df["date"] + pd.Timedelta(days=1)
    df["day_of_week"] = df["target_date"].dt.day_name().str[:3]
    next_imp = g["impressions"].shift(-1)
    next_clk = g["clicks"].shift(-1)
    df["next_ctr"] = 100 * next_clk / next_imp
    labelled = next_imp >= config.MIN_IMPRESSIONS
    df[config.TARGET] = np.where(labelled, (df["next_ctr"] > df["benchmark_ctr"]).astype(float), np.nan)

    has_history = imp_7 > 0
    is_last_day = next_imp.isna()
    keep = has_history & (labelled | (include_unlabelled & is_last_day))
    cols = (["campaign_id", "date", "target_date"] + config.FEATURES
            + ["ctr_1d", "ctr_7d", "benchmark_ctr", "next_ctr", config.TARGET])
    return df.loc[keep, cols].reset_index(drop=True)


def assign_split(features: pd.DataFrame, train_end, val_end) -> pd.Series:
    return pd.Series(np.select([features["target_date"] <= train_end, features["target_date"] <= val_end],
                               ["train", "val"], "test"), index=features.index)


def prepare_dataset(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full preprocessing chain: split dates -> benchmark -> features -> split labels."""
    train_end, val_end = split_dates(panel)
    benchmark = compute_benchmark(panel, train_end)
    feats = build_features(panel, benchmark)
    feats[config.TARGET] = feats[config.TARGET].astype(int)
    feats["split"] = assign_split(feats, train_end, val_end)
    meta = {
        "train_end": str(train_end.date()),
        "val_end": str(val_end.date()),
        "benchmark_ctr_pct": benchmark,
        "benchmark_group": config.BENCHMARK_GROUP,
        "min_impressions": config.MIN_IMPRESSIONS,
        "date_range": [str(panel["date"].min().date()), str(panel["date"].max().date())],
    }
    return feats, meta

"""Feature pipeline: reproducible, chronological, correctly labelled and free of look-ahead."""

import numpy as np
import pandas as pd
import pytest

from src import config, features


def test_preprocessing_reproduces_the_saved_dataset(panel, dataset, meta):
    rebuilt, rebuilt_meta = features.prepare_dataset(panel)
    assert (rebuilt_meta["train_end"], rebuilt_meta["val_end"]) == (meta["train_end"], meta["val_end"])
    assert rebuilt_meta["benchmark_ctr_pct"] == pytest.approx(meta["benchmark_ctr_pct"])
    pd.testing.assert_frame_equal(rebuilt, dataset, check_dtype=False, check_exact=False, rtol=1e-9)


def test_split_is_chronological(dataset):
    dates = dataset.groupby("split")["target_date"].agg(["min", "max"])
    assert dates.loc["train", "max"] < dates.loc["val", "min"]
    assert dates.loc["val", "max"] < dates.loc["test", "min"]


def test_benchmark_ignores_data_after_training(panel, meta):
    train_end = pd.Timestamp(meta["train_end"])
    later = panel["date"] > train_end
    distorted = panel.copy()
    distorted.loc[later, "clicks"] = distorted.loc[later, "impressions"]  # CTR = 100% after training
    assert features.compute_benchmark(distorted, train_end) == meta["benchmark_ctr_pct"]


def test_labels_match_next_day_ctr(dataset):
    expected = (dataset["next_ctr"] > dataset["benchmark_ctr"]).astype(int)
    assert (dataset[config.TARGET] == expected).all()


@pytest.mark.parametrize("position", [0.3, 0.6, 0.9])
def test_features_use_only_past_data(panel, meta, position):
    """Features for day t must not change when every day after t is removed from the data."""
    benchmark = meta["benchmark_ctr_pct"]
    full = features.build_features(panel, benchmark)
    days = np.sort(full["date"].unique())
    day = days[int(position * len(days))]
    seen_then = features.build_features(panel[panel["date"] <= day], benchmark, include_unlabelled=True)
    full = full[full["date"] == day].set_index("campaign_id")[config.FEATURES]
    seen_then = seen_then[seen_then["date"] == day].set_index("campaign_id")[config.FEATURES]
    assert len(full) > 0
    pd.testing.assert_frame_equal(seen_then.loc[full.index], full, check_dtype=False)

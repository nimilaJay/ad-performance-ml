"""Data loading and cleaning: export parsing, invalid rows, duplicates and calendar gaps."""

import pandas as pd

from src import data


def test_clean_drops_invalid_merges_duplicates_and_fills_gaps():
    raw = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-03", "2025-01-05", "2025-01-04"],
        "campaign_id": ["a"] * 6, "platform": ["Meta Ads"] * 6, "objective": ["Traffic"] * 6,
        "spend": [10, 5, 8, -1, 7, 6],
        "impressions": [1000, 500, 900, 100, 800, 50],
        "clicks": [10, 5, 9, 1, 8, 60],  # 2025-01-04 has more clicks than impressions
    })
    out = data.clean(raw, verbose=False)
    log = out.attrs["cleaning_log"]

    assert log["invalid_values"] == 2
    assert log["duplicates_merged"] == 1
    assert log["gap_days_filled"] == 2
    assert out["date"].dt.strftime("%m-%d").tolist() == ["01-01", "01-02", "01-03", "01-04", "01-05"]
    assert out["impressions"].tolist() == [1500, 900, 0, 0, 800]
    assert out["spend"].tolist() == [15, 8, 0, 0, 7]


def test_read_export_handles_google_ads_layout(tmp_path):
    export = tmp_path / "google_ads_export.csv"
    export.write_text(
        "Campaign performance report\n"
        '"1 January 2025 - 2 January 2025"\n'
        "Day,Campaign,Campaign type,Cost,Impr.,Clicks\n"
        '2025-01-01,Spring sale,Search,"$1,204.50","12,000",340\n'
        "2025-01-02,Spring sale,Search,--,0,0\n"
        "Total,,,1204.50,12000,340\n"
    )
    df = data.read_export(export, platform="Google Ads")

    assert list(df.columns) == data.REQUIRED_COLUMNS
    assert len(df) == 2  # the "Total" line is dropped
    assert df["spend"].tolist() == [1204.50, 0.0]
    assert df["impressions"].tolist() == [12000, 0]
    assert (df["platform"] == "Google Ads").all()


def test_anonymised_ids_are_stable_and_hide_names():
    a = data.anonymise_id("Google Ads|Spring sale")
    assert a == data.anonymise_id("Google Ads|Spring sale")
    assert a.startswith("cmp_") and "Spring" not in a

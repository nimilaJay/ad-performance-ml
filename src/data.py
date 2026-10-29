"""Loading, validating and cleaning campaign-day data.

The pipeline expects one row per campaign per day with these columns:

    date, campaign_id, platform, objective, spend, impressions, clicks

Daily campaign reports exported from Google Ads or Meta Ads Manager can be converted into this
format with the command-line tool at the bottom of this file, e.g.

    python -m src.data --export google_ads.csv "Google Ads" --export meta_ads.csv "Meta Ads"
"""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

REQUIRED_COLUMNS = ["date", "campaign_id", "platform", "objective", "spend", "impressions", "clicks"]

# Column names used by common ad-platform exports, mapped to the standard schema (matched
# case-insensitively). Extend this if your export uses different headers.
COLUMN_ALIASES = {
    "date": ["date", "day", "reporting starts", "reporting_start", "date_start"],
    "campaign_id": ["campaign_id", "campaign id", "campaign", "campaign name", "campaign_name"],
    "platform": ["platform", "ad platform", "network", "publisher platform"],
    "objective": ["objective", "campaign objective", "campaign type", "advertising channel type",
                  "campaign goal"],
    "spend": ["spend", "cost", "amount spent", "amount spent (usd)", "amount spent (gbp)",
              "amount spent (eur)", "amount spent (lkr)", "amount_spent"],
    "impressions": ["impressions", "impr.", "impr"],
    "clicks": ["clicks", "link clicks", "clicks (all)", "inline_link_clicks"],
}


def _find_header_row(path: Path, max_lines: int = 10) -> int:
    """Google Ads exports start with a title and date-range line before the header row."""
    date_names = set(COLUMN_ALIASES["date"])
    with open(path, newline="", encoding="utf-8-sig") as f:
        for i, row in enumerate(csv.reader(f)):
            if i >= max_lines:
                break
            if any(cell.strip().lower() in date_names for cell in row):
                return i
    return 0


def _to_number(s: pd.Series) -> pd.Series:
    """Parse numbers that may contain currency symbols, thousands separators or '--'."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    cleaned = s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True).replace({"": np.nan, "-": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {c.strip().lower(): c for c in df.columns}
    rename = {}
    for std, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lookup:
                rename[lookup[alias]] = std
                break
    return df.rename(columns=rename)


def anonymise_id(value: str) -> str:
    """One-way hash so campaign names (which can be commercially sensitive) never leave the raw export."""
    return "cmp_" + hashlib.sha256(str(value).encode()).hexdigest()[:10]


def read_export(path, platform: str | None = None, objective: str | None = None) -> pd.DataFrame:
    """Read one platform export and return it in the standard schema."""
    path = Path(path)
    df = pd.read_csv(path, skiprows=_find_header_row(path), encoding="utf-8-sig")
    df = standardise_columns(df)
    if platform is not None:
        df["platform"] = platform
    if objective is not None and "objective" not in df.columns:
        df["objective"] = objective
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing columns {missing}. Found: {list(df.columns)}")
    df = df[REQUIRED_COLUMNS].copy()
    # Drop "Total" summary lines and parse "$1,204.50"-style numbers so the combined CSV is clean.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["date"] = df["date"].dt.date
    for col in ["spend", "impressions", "clicks"]:
        df[col] = _to_number(df[col]).fillna(0)
    return df


def clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Validate types, drop invalid rows and fill calendar gaps within each campaign.

    Returns a daily panel: every campaign has one row per calendar day between its first and last
    observed day. Days with no delivery (paused or missing) have zero spend/impressions/clicks.
    """
    log = {"rows_in": len(df)}
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["spend", "impressions", "clicks"]:
        df[col] = _to_number(df[col])
    for col in ["campaign_id", "platform", "objective"]:
        df[col] = df[col].astype(str).str.strip()

    # Totals/summary lines in exports have no parseable date.
    df = df.dropna(subset=["date"])
    log["no_date"] = log["rows_in"] - len(df)

    df[["spend", "impressions", "clicks"]] = df[["spend", "impressions", "clicks"]].fillna(0)
    invalid = (df[["spend", "impressions", "clicks"]] < 0).any(axis=1) | (df["clicks"] > df["impressions"])
    log["invalid_values"] = int(invalid.sum())
    df = df[~invalid]

    # Duplicate campaign-days (e.g. overlapping exports) are aggregated.
    before = len(df)
    df = (df.groupby(["campaign_id", "date"], as_index=False)
            .agg(platform=("platform", "first"), objective=("objective", "first"),
                 spend=("spend", "sum"), impressions=("impressions", "sum"), clicks=("clicks", "sum")))
    log["duplicates_merged"] = before - len(df)

    panels = []
    for cid, g in df.groupby("campaign_id"):
        full = pd.date_range(g["date"].min(), g["date"].max(), freq="D")
        g = g.set_index("date").reindex(full)
        g.index.name = "date"
        g["campaign_id"] = cid
        g[["platform", "objective"]] = g[["platform", "objective"]].ffill()
        g[["spend", "impressions", "clicks"]] = g[["spend", "impressions", "clicks"]].fillna(0)
        panels.append(g.reset_index())
    out = pd.concat(panels, ignore_index=True)
    log["gap_days_filled"] = len(out) - len(df)
    log["rows_out"] = len(out)
    log["campaigns"] = out["campaign_id"].nunique()

    if verbose:
        print("Cleaning summary:", log)
    out = out.sort_values(["campaign_id", "date"]).reset_index(drop=True)
    out.attrs["cleaning_log"] = log
    return out


def load(path=None, verbose: bool = True) -> pd.DataFrame:
    """Load the standard-schema campaign CSV used for modelling."""
    if path is None:
        path = config.DATA_PATH
    path = Path(path)
    if verbose:
        print(f"Loading {path}")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing {missing}; convert exports with `python -m src.data`.")
    out = clean(df[REQUIRED_COLUMNS], verbose=verbose)
    out.attrs["source_path"] = str(path)
    return out


def main():
    parser = argparse.ArgumentParser(description="Combine ad-platform exports into the standard schema.")
    parser.add_argument("--export", nargs=2, action="append", metavar=("CSV", "PLATFORM"), required=True,
                        help='An export file and its platform name, e.g. --export meta.csv "Meta Ads"')
    parser.add_argument("--output", default=str(config.DATA_PATH))
    parser.add_argument("--keep-names", action="store_true",
                        help="Keep original campaign names instead of anonymised IDs")
    args = parser.parse_args()

    frames = [read_export(p, platform=name) for p, name in args.export]
    df = pd.concat(frames, ignore_index=True)
    if not args.keep_names:
        df["campaign_id"] = (df["platform"] + "|" + df["campaign_id"].astype(str)).map(anonymise_id)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows from {len(frames)} export(s) to {args.output}")


if __name__ == "__main__":
    main()

"""Central configuration: paths, split settings, benchmark definition and feature lists."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_PATH = DATA_RAW / "campaign_daily.csv"

RANDOM_STATE = 42

# Chronological split on the *target* date: oldest 70% train, next 15% validation, newest 15% test.
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# A campaign-day needs at least this many impressions for its CTR to count as a reliable label.
MIN_IMPRESSIONS = 100

# Benchmark definition. "train_median" = median daily CTR per platform x objective, computed on the
# training period only (no look-ahead). Grouping by objective compares like with like: awareness
# campaigns structurally get far lower CTRs than traffic or lead campaigns. Alternatively, give
# fixed CTR values (in %) keyed "<platform> | <objective>", e.g. {"Meta Ads | Traffic": 0.90, ...}.
BENCHMARK_MODE = "train_median"
BENCHMARK_GROUP = ["platform", "objective"]

TARGET = "above_benchmark_next_day"

CATEGORICAL_FEATURES = ["platform", "objective", "day_of_week"]
NUMERIC_FEATURES = [
    "campaign_day",
    "spend_1d",
    "spend_7d_avg",
    "spend_cum",
    "impressions_1d",
    "impressions_7d_avg",
    "clicks_1d",
    "clicks_7d_avg",
    "ctr_index_1d",
    "ctr_index_3d",
    "ctr_index_7d",
    "ctr_trend",
    "cpm_7d",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

# Human-readable labels used in plots and the dashboard.
FEATURE_LABELS = {
    "platform": "Advertising platform",
    "objective": "Campaign objective",
    "day_of_week": "Day of week (next day)",
    "campaign_day": "Campaign age (days)",
    "spend_1d": "Spend, last day",
    "spend_7d_avg": "Avg daily spend, last 7 days",
    "spend_cum": "Cumulative spend",
    "impressions_1d": "Impressions, last day",
    "impressions_7d_avg": "Avg daily impressions, last 7 days",
    "clicks_1d": "Clicks, last day",
    "clicks_7d_avg": "Avg daily clicks, last 7 days",
    "ctr_index_1d": "CTR index, last day",
    "ctr_index_3d": "CTR index, last 3 days",
    "ctr_index_7d": "CTR index, last 7 days",
    "ctr_trend": "CTR trend (1d minus 7d index)",
    "cpm_7d": "CPM, last 7 days",
}

for _d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

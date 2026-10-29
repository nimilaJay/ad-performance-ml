"""Build a campaign-day dataset with platform, objective and delivery patterns."""

import argparse

import numpy as np
import pandas as pd

from src import config

# Typical daily CTR (%) by platform and objective.
BASE_CTR = {
    "Google Ads": {"Awareness": 0.6, "Traffic": 3.4, "Engagement": 1.6, "Leads": 4.4, "Sales": 3.7},
    "Meta Ads": {"Awareness": 0.55, "Traffic": 1.5, "Engagement": 1.3, "Leads": 1.0, "Sales": 0.95},
}
# Typical CPM (cost per 1,000 impressions).
BASE_CPM = {
    "Google Ads": {"Awareness": 3.0, "Traffic": 28.0, "Engagement": 8.0, "Leads": 45.0, "Sales": 38.0},
    "Meta Ads": {"Awareness": 5.0, "Traffic": 9.0, "Engagement": 7.0, "Leads": 14.0, "Sales": 15.0},
}
OBJECTIVE_MIX = {
    "Google Ads": [0.15, 0.25, 0.10, 0.25, 0.25],
    "Meta Ads": [0.20, 0.25, 0.20, 0.15, 0.20],
}
DOW_EFFECT = {  # Monday=0
    "Google Ads": [1.06, 1.07, 1.05, 1.03, 0.98, 0.88, 0.90],
    "Meta Ads": [0.95, 0.96, 0.98, 1.00, 1.03, 1.07, 1.06],
}


def generate_campaign(rng, cid, platform, objective, start, n_days):
    quality = rng.lognormal(0.0, 0.22)            # creative / targeting quality
    budget = rng.lognormal(np.log(80), 0.7)        # usual daily budget
    fatigue_rate = rng.uniform(0.006, 0.014) if platform == "Meta Ads" else rng.uniform(0.001, 0.005)
    learning_days = 5 if platform == "Meta Ads" else 3
    base_ctr = BASE_CTR[platform][objective] / 100
    base_cpm = BASE_CPM[platform][objective]

    rows, noise, age_since_refresh, budget_level = [], 0.0, 0, budget
    for d in range(n_days):
        date = start + pd.Timedelta(days=d)
        dow = date.day_of_week

        if rng.random() < 0.03:                     # budget changes
            budget_level = budget * rng.choice([0.5, 0.75, 1.5, 2.0, 3.0])
        if rng.random() < 0.015 and d > 14:         # creative refresh resets fatigue
            age_since_refresh = 0
            quality *= rng.lognormal(0, 0.15)
        if rng.random() < 0.02:                     # paused day: no delivery, no row exported
            age_since_refresh += 1
            continue

        spend = budget_level * rng.uniform(0.80, 1.05)
        scale = spend / budget                      # >1 means spending above usual budget
        cpm = base_cpm * scale ** 0.25 * rng.lognormal(0, 0.10)
        impressions = int(spend / cpm * 1000)

        fatigue = max(0.45, np.exp(-fatigue_rate * age_since_refresh))
        learning = 0.80 if d < learning_days else 1.0
        # Persistent drift (audience/competition changes) plus a one-day auction shock.
        noise = 0.75 * noise + rng.normal(0, 0.12 if d >= learning_days else 0.25)
        shock = rng.normal(0, 0.12)
        ctr = (base_ctr * quality * fatigue * learning * DOW_EFFECT[platform][dow]
               * scale ** -0.25 * np.exp(noise + shock))
        ctr = float(np.clip(ctr, 0.0005, 0.25))
        clicks = int(rng.binomial(impressions, ctr)) if impressions > 0 else 0

        rows.append({"date": date.date().isoformat(), "campaign_id": cid, "platform": platform,
                     "objective": objective, "spend": round(spend, 2),
                     "impressions": impressions, "clicks": clicks})
        age_since_refresh += 1
    return rows


def generate(n_campaigns=90, start="2025-01-01", end="2025-10-31", seed=config.RANDOM_STATE):
    rng = np.random.default_rng(seed)
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    span = (end - start).days
    rows = []
    for i in range(n_campaigns):
        platform = "Google Ads" if rng.random() < 0.55 else "Meta Ads"
        objective = rng.choice(list(BASE_CTR[platform]), p=OBJECTIVE_MIX[platform])
        n_days = int(rng.integers(21, 100))
        offset = int(rng.integers(0, max(1, span - 21)))
        n_days = min(n_days, span - offset)
        rows += generate_campaign(rng, f"cmp_{i:03d}", platform, objective,
                                  start + pd.Timedelta(days=offset), n_days)
    df = pd.DataFrame(rows)

    # A few duplicated rows, as happens with overlapping report exports.
    dupes = df.sample(frac=0.005, random_state=seed)
    return pd.concat([df, dupes]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaigns", type=int, default=90)
    parser.add_argument("--output", default=str(config.DATA_PATH))
    args = parser.parse_args()
    df = generate(n_campaigns=args.campaigns)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows ({df['campaign_id'].nunique()} campaigns) to {args.output}")


if __name__ == "__main__":
    main()

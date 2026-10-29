# Explainable ML for Digital Marketing Campaign Performance Prediction

Predicts whether a digital advertising campaign's **click-through rate (CTR) tomorrow will beat its benchmark**, using
an **Explainable Boosting Machine (EBM)**, and explains every prediction. Logistic Regression and CatBoost are trained
as comparison models, alongside a naive "tomorrow = today" persistence baseline. A Streamlit dashboard lets marketing
teams enter recent campaign data and see the prediction, its probability and the factors driving it.

**Live dashboard:** https://ctr-dashboard.purplefield-4d57f34d.eastasia.azurecontainerapps.io (Azure Container Apps; the first visit after a quiet period takes about 20 seconds while it starts).

## Project structure

```
├── app/
│   ├── streamlit_app.py      Dashboard: predict, watchlist, model insights, performance, data overview
│   └── charts.py             Plotly charts (light/dark aware)
├── src/
│   ├── config.py             Paths, split fractions, benchmark definition, feature lists
│   ├── data.py               Load/validate/clean data; convert Google Ads / Meta Ads exports
│   ├── generate_data.py      Campaign dataset generation utility
│   ├── features.py           Feature engineering, benchmark, target, chronological split
│   ├── models.py             EBM / Logistic Regression / CatBoost definitions and search grids
│   ├── train.py              Tuning, threshold selection, refit, test evaluation, plots
│   ├── evaluate.py           Metrics, bootstrap confidence intervals, evaluation plots
│   ├── explain.py            EBM global + local explanations, PDPs, permutation importance
│   └── plotting.py           Shared figure style and colour palette
├── notebooks/01_eda.ipynb    Exploratory data analysis
├── report/                   269122F_Jayawardena_ML_Report.pdf, its build script and dashboard screenshots
├── data/raw/                 Input data (campaign_daily.csv)
├── data/processed/           Modelling dataset written by train.py
├── models/                   Trained models + metadata.json
├── reports/                  Metrics tables, predictions, explanation summaries, figures/
├── tests/                    Data, feature-pipeline, model-quality and dashboard tests (pytest)
├── Dockerfile                Container image for the dashboard
├── .github/workflows/        GitHub Actions: CI/CD to Azure, retraining, daily health check
└── deploy/azure_setup.sh     One-time Azure setup for the pipeline
```

## Setup

Python 3.12 is recommended (all dependencies have wheels for it).

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the pipeline

```bash
python -m src.train          # preprocess, tune, evaluate  -> models/, reports/, reports/figures/
python -m src.explain        # explainability analysis      -> reports/, reports/figures/
jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb   # EDA (or open it in Jupyter)
streamlit run app/streamlit_app.py                                           # dashboard
python report/build_report.py                                                # rebuild report/269122F_Jayawardena_ML_Report.pdf
pytest                                                                       # tests, including the model quality gate
```

`src.train` uses `data/raw/campaign_daily.csv` by default. Use `--data path.csv` to select another input file.

## Deployment and MLOps (Azure)

The dashboard runs on **Azure Container Apps**. Everything after a one-time setup happens in GitHub Actions:

```
Retrain workflow ("Run workflow" button)
  train + tune -> explainability -> tests + quality gate -> branch + one-click pull request with a metrics comparison
                                                                       |
                                                                       | merge
                                                                       v
CI/CD workflow (every push to main)
  1. test    pytest: cleaning, no look-ahead in features, models reproduce the reported results,
             EBM explanations add up, quality gate, dashboard renders
  2. build   Docker image, smoke-tested as a running container
  3. deploy  image tagged with the commit ID -> Azure Container Registry
             -> Azure Container Apps (new revision) -> live check on the public HTTPS URL

Health check workflow (daily): the live app must answer, otherwise GitHub emails a failure
```

| MLOps practice | How it is done here |
|---|---|
| Versioning | Code, data, models and metrics live in git; every image tag and Container Apps revision carries its commit, and the dashboard sidebar shows it as "App build". |
| Continuous integration | `tests/`: data cleaning, chronological split, no look-ahead in features, models reproduce the reported results, EBM explanations add up exactly, dashboard renders. |
| Quality gate | A release is blocked unless the EBM reaches a test ROC-AUC of at least 0.80 and beats the persistence baseline (`tests/test_models.py`). |
| Continuous delivery | `.github/workflows/ci-cd.yml`: test -> build -> deploy, with the `production` environment showing the live URL. |
| Continuous training | `.github/workflows/retrain.yml`: retrains on the data in the repository, pushes the models to a branch and links a pull request with a metrics comparison; a person reviews and merges it, which deploys. |
| Monitoring | `.github/workflows/health-check.yml` daily, plus a live check after every deploy; logs and metrics in the Azure portal. |
| Rollback | Re-run the **deploy** job of an earlier successful CI/CD run: that commit's image goes live again. |
| Security | GitHub signs in to Azure with OIDC (no stored password) as a managed identity limited to this resource group; the app pulls images with its own pull-only identity. |

**One-time setup** (needs the Azure CLI, Docker and the GitHub CLI): `az login`, then `bash deploy/azure_setup.sh`.
It creates the registry, the app and the GitHub connection, and prints the live URL. Azure for Students allows
only a few regions and one Container Apps environment per subscription; the script defaults to East Asia. The app scales to zero when idle,
so it stays within the Container Apps free monthly allowance; the Basic registry costs about US$5 a month. Remove
everything with `az group delete --name rg-ctr-dashboard --yes`.

The app also runs on [Streamlit Community Cloud](https://share.streamlit.io) without Docker (main file
`app/streamlit_app.py`, Python 3.12).

## Collecting the data

Only **aggregated, campaign-level** data is used: no user-level or personal data. Use accounts you are authorised to
access and keep client names confidential (the converter replaces campaign names with one-way hashed IDs by default).

**Google Ads:** Campaigns → Reports → a predefined *Campaign* report (or a custom one), segmented by **Day**, with
the columns *Day, Campaign, Campaign type, Cost, Impr., Clicks*. Download as CSV.

**Meta Ads Manager:** Campaigns tab → Breakdown → By time → **Day**. Customise columns to include *Campaign name,
Objective, Amount spent, Impressions, Link clicks* (and the date columns), then Export table data as CSV.

Then convert and combine the exports into the standard schema:

```bash
python -m src.data --export google_ads.csv "Google Ads" --export meta_ads.csv "Meta Ads"
# -> data/raw/campaign_daily.csv with: date, campaign_id, platform, objective, spend, impressions, clicks
```

The converter handles the title lines Google Ads puts above the header, *Total* rows, currency symbols and thousands
separators. If your export uses other column names, add them to `COLUMN_ALIASES` in `src/data.py`. Notes:

* Use **one click definition per platform** (e.g. Meta *Link clicks* throughout), or CTRs are not comparable over time.
* Objective labels need not match across platforms: benchmarks are computed per *platform × objective*.
* Every *platform × objective* group needs enough history in the training period to give a stable benchmark;
  merge or drop very small groups.
* Target size: roughly 1,000+ labelled campaign-days (a campaign-day counts if the next day has ≥ 100 impressions).

## Method

**Unit of analysis.** One row per campaign per day *t*, built only from information available at the end of day *t*.

**Target.** `above_benchmark_next_day = 1` if CTR on day *t+1* exceeds the benchmark for the campaign's platform and
objective. The benchmark is the **median daily CTR of that group during the training period**, so it is fixed before
validation/test and leaks no future information. Days with fewer than 100 impressions are not labelled (CTR too noisy).

**Features** (`src/features.py`):

| Group | Features |
|---|---|
| Campaign context | platform, objective, day of week (of the target day), campaign age in days |
| Spend | last-day spend, 7-day average daily spend, cumulative spend, 7-day CPM |
| Delivery volume | last-day and 7-day average impressions and clicks |
| CTR history | CTR index (CTR ÷ benchmark) over the last 1, 3 and 7 days; trend (1-day minus 7-day index) |

**Chronological split.** Rows are ordered by target date: oldest 70% train, next 15% validation, newest 15% test.
No shuffling, so the model is always evaluated on days later than anything it was trained on.

**Training** (`src/train.py`). Each model is grid-searched on the training period and scored by validation ROC-AUC.
The decision threshold is chosen on validation by Youden's J (balancing detection of above- and below-benchmark days).
The best configuration is refit on train + validation and evaluated **once** on the test set, reporting precision,
recall, F1, specificity (*at-risk recall*), balanced accuracy, ROC-AUC, PR-AUC and Brier score. 95% confidence
intervals come from a campaign-level (cluster) bootstrap, and performance is also broken down by platform and objective.

**Explainability** (`src/explain.py`). EBM is a glass-box generalised additive model with pairwise interactions, so its
explanations are the model itself rather than post-hoc approximations:

* global term importance and learned shape functions for each feature, with bagging uncertainty bands;
* the strongest pairwise interaction as a heatmap;
* exact local explanations (intercept + per-term contributions = predicted log-odds) for individual campaign-days;
* Partial Dependence Plots comparing EBM, Logistic Regression and CatBoost;
* permutation importance on the test set for all three models.

## Outputs

| File | Contents |
|---|---|
| `reports/model_comparison.csv`, `reports/metrics.json` | Validation and test metrics (with CIs) for every model |
| `reports/tuning_results.csv` | Every hyperparameter configuration tried and its validation ROC-AUC |
| `reports/segment_performance.csv` | Test performance per platform and per objective |
| `reports/test_predictions.csv` | Test-set probabilities from every model |
| `reports/ebm_term_importance.csv`, `reports/permutation_importance.csv` | Importance tables |
| `reports/explainability_summary.json` | Key numbers behind the explanation figures |
| `reports/figures/` | ROC/PR/calibration curves, confusion matrices, EBM shapes, PDPs, local explanations, EDA plots |

## Limitations to keep in mind

Predictions are **associations, not causal effects**: for example, the model can say campaigns that recently raised
spend tend to see lower CTR, not that raising spend causes it. Benchmarks depend on the campaigns in the training
period; platform algorithm changes, seasonality and tracking changes cause drift, so the model should be retrained
regularly and its validation-vs-test performance monitored.

"""Campaign CTR Predictor: interactive dashboard for the EBM campaign-performance model.

Run from the project root:  streamlit run app/streamlit_app.py
"""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.charts import MODEL_ORDER, Charts  # noqa: E402
from src import config, data, explain, features  # noqa: E402

st.set_page_config(page_title="Campaign CTR Predictor", page_icon="📈", layout="wide")


# ---------------------------------------------------------------- loading

@st.cache_resource
def load_artifacts():
    if not (config.MODELS_DIR / "metadata.json").exists():
        st.error("No trained models found. Run `python -m src.train` and `python -m src.explain` first.")
        st.stop()
    return explain.load_artifacts()


@st.cache_data
def load_panel(path):
    return data.load(path, verbose=False)


@st.cache_data
def load_reports():
    r = config.REPORTS_DIR
    with open(r / "metrics.json") as f:
        metrics = json.load(f)
    return (pd.read_csv(r / "model_comparison.csv"), pd.read_csv(r / "test_predictions.csv"),
            pd.read_csv(r / "tuning_results.csv"), metrics)


@st.cache_data
def cached_pdp(_models, X, feature):
    grid = explain.feature_grid(X, feature, n=25)
    return grid, {name: explain.partial_dependence(_models[name], X, feature, grid) for name in MODEL_ORDER}


models, meta, dataset = load_artifacts()
ebm = models["EBM"]
benchmark = meta["benchmark_ctr_pct"]
threshold = meta["thresholds"]["EBM"]
panel = load_panel(str(config.ROOT / meta["data_path"]))  # relative paths resolve against the project root

try:
    theme_mode = st.context.theme.type or "light"
except AttributeError:
    theme_mode = "light"
charts = Charts(theme_mode)


def bench_keys(df):
    return df[meta["benchmark_group"]].astype(str).agg(" | ".join, axis=1)


def fmt_pct(x, digits=2):
    return "–" if pd.isna(x) else f"{x:.{digits}f}%"


# ---------------------------------------------------------------- header & sidebar

st.title("Campaign CTR Predictor")
st.caption("Predicts whether a digital ad campaign's click-through rate (CTR) tomorrow will beat its "
           "benchmark, and explains why, using an Explainable Boosting Machine (EBM).")
if meta.get("is_synthetic"):
    st.warning("**Synthetic placeholder data.** These models were trained on simulated campaigns for "
               "development only. Retrain on real Google Ads / Meta Ads exports before drawing conclusions.",
               icon="⚠️")

with st.sidebar:
    st.subheader("How it works")
    st.markdown(
        "Each prediction uses the campaign's **last 7 days** of spend, impressions and clicks, plus its "
        "platform, objective and age. The target is whether **tomorrow's CTR** exceeds the benchmark: the "
        "median daily CTR of comparable campaigns (same platform and objective) during the training period.")
    st.markdown(f"**Decision threshold:** {threshold:.2f} (chosen on validation data)")
    st.subheader("Benchmarks (CTR)")
    st.dataframe(pd.DataFrame({"Group": list(benchmark), "CTR": [fmt_pct(v) for v in benchmark.values()]}),
                 hide_index=True, width="stretch")
    st.caption(f"Data: {meta['date_range'][0]} to {meta['date_range'][1]} · "
               f"train ≤ {meta['train_end']} < validation ≤ {meta['val_end']} < test · "
               f"models trained {meta['trained_at'][:10]}")
    # APP_VERSION is the git commit the Docker image was built from (set by the CI/CD pipeline).
    st.caption(f"App build: {os.environ.get('APP_VERSION', 'local')[:7]}")
    st.caption("Predictions describe statistical associations in historical data, not causal effects.")

tab_predict, tab_watch, tab_learned, tab_perf, tab_data = st.tabs(
    ["Predict", "Campaign watchlist", "What the model learned", "Model performance", "Data overview"])


# ---------------------------------------------------------------- predict

def explain_text(contrib):
    def items(d):
        return "; ".join(f"{lab} = {val} ({c:+.2f})" for lab, val, c in
                         zip(d["label"], d["value"], d["contribution"]))
    ups = contrib[contrib["contribution"] > 0].head(3)
    downs = contrib[contrib["contribution"] < 0].head(3)
    lines = []
    if len(ups):
        lines.append(f"**Raising the chance:** {items(ups)}")
    if len(downs):
        lines.append(f"**Lowering the chance:** {items(downs)}")
    return "\n\n".join(lines)


def render_predict():
    source = st.radio("Start from", ["An existing campaign in the dataset", "A blank campaign"], horizontal=True)
    combos = dataset[["platform", "objective"]].drop_duplicates()
    platforms = sorted(combos["platform"])

    actual_next = None
    if source.startswith("An existing"):
        c1, c2 = st.columns([1, 2])
        cid = c1.selectbox("Campaign", sorted(panel["campaign_id"].unique()))
        camp = panel[panel["campaign_id"] == cid].reset_index(drop=True)
        dates = camp["date"].dt.date.tolist()
        default = next((d for d in dates if str(d) > meta["val_end"]), dates[-1])
        end = c2.select_slider("Last day of data (predict the day after)", options=dates, value=default)
        upto = camp[camp["date"].dt.date <= end]
        hist = upto.tail(7)[["date", "spend", "impressions", "clicks"]].reset_index(drop=True)
        d_platform, d_objective = camp["platform"].iat[0], camp["objective"].iat[0]
        d_age, d_cum = len(upto), float(upto["spend"].sum())
        nxt = camp[camp["date"].dt.date == end + pd.Timedelta(days=1)]
        if len(nxt) and nxt["impressions"].iat[0] > 0:
            actual_next = 100 * nxt["clicks"].iat[0] / nxt["impressions"].iat[0]
        key = f"{cid}_{end}"
        if str(end) < meta["val_end"]:
            st.caption(f"ℹ️ This day is inside the training/validation period (≤ {meta['val_end']}), so the "
                       "model has seen its outcome. Pick a later day for a genuine out-of-sample check.")
    else:
        hist = None  # built below, once platform and objective are chosen
        d_platform, d_objective, d_age, d_cum = platforms[0], None, 21, 120.0 * 21
        key = "blank"

    st.markdown("##### Campaign details")
    c1, c2, c3, c4 = st.columns(4)
    platform = c1.selectbox("Advertising platform", platforms, index=platforms.index(d_platform),
                            key=f"platform_{key}")
    objectives = sorted(combos.loc[combos["platform"] == platform, "objective"])
    objective = c2.selectbox("Campaign objective", objectives,
                             index=objectives.index(d_objective) if d_objective in objectives else 0,
                             key=f"objective_{key}")
    age = c3.number_input("Campaign age on last day (days)", min_value=1, value=int(d_age), key=f"age_{key}")
    cum = c4.number_input("Total spend to date", min_value=0.0, value=round(d_cum, 2), step=50.0,
                          key=f"cum_{key}")

    if hist is None:
        # Example history: CTR sliding from 15% above to 15% below this group's benchmark (creative fatigue).
        bench_ctr = benchmark[bench_keys(pd.DataFrame({"platform": [platform], "objective": [objective]})).iat[0]]
        last = pd.Timestamp.today().normalize() - pd.Timedelta(days=1)
        hist = pd.DataFrame({"date": pd.date_range(end=last, periods=7, freq="D"), "spend": [120.0] * 7,
                             "impressions": [9000] * 7,
                             "clicks": [round(9000 * bench_ctr / 100 * f) for f in np.linspace(1.15, 0.85, 7)]})
        key = f"blank_{platform}_{objective}"

    st.markdown("##### Last 7 days of delivery")
    st.caption("Edit any value to explore a what-if scenario, e.g. a budget increase or a CTR drop.")
    hist = hist.assign(date=pd.to_datetime(hist["date"]).dt.date)
    edited = st.data_editor(
        hist, key=f"editor_{key}", hide_index=True, num_rows="fixed", disabled=["date"], width="stretch",
        column_config={
            "date": st.column_config.DateColumn("Date"),
            "spend": st.column_config.NumberColumn("Spend", min_value=0.0, format="%.2f"),
            "impressions": st.column_config.NumberColumn("Impressions", min_value=0, step=1),
            "clicks": st.column_config.NumberColumn("Clicks", min_value=0, step=1),
        })

    errors = []
    if (edited["clicks"] > edited["impressions"]).any():
        errors.append("Clicks cannot exceed impressions.")
    if edited["impressions"].sum() == 0:
        errors.append("At least one day needs impressions.")
    if errors:
        for e in errors:
            st.error(e, icon="🚫")
        return

    one = edited.assign(date=pd.to_datetime(edited["date"]), campaign_id="input",
                        platform=platform, objective=objective)
    row = features.build_features(one, benchmark, include_unlabelled=True).tail(1).copy()
    row["campaign_day"] = age
    row["spend_cum"] = max(cum, float(edited["spend"].sum()))
    X = row[config.FEATURES]
    probs = {name: float(models[name].predict_proba(X)[:, 1][0]) for name in MODEL_ORDER}
    p = probs["EBM"]
    contrib, intercept = explain.ebm_local(ebm, X)
    bench = row["benchmark_ctr"].iat[0]

    st.divider()
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.metric("Chance tomorrow's CTR beats the benchmark", f"{p:.0%}")
        if p >= threshold:
            st.success("**On track:** likely to beat the benchmark tomorrow.", icon="✅")
        else:
            st.error("**At risk:** likely to fall below the benchmark tomorrow.", icon="⚠️")
        st.markdown(
            f"- Benchmark ({platform}, {objective}): **{fmt_pct(bench)}**\n"
            f"- CTR last day: **{fmt_pct(row['ctr_1d'].iat[0])}** · last 7 days: **{fmt_pct(row['ctr_7d'].iat[0])}**\n"
            f"- Predicting for: **{row['target_date'].iat[0]:%a %d %b %Y}**")
        if actual_next is not None:
            outcome = "above" if actual_next > bench else "below"
            st.info(f"What actually happened (original data): CTR {actual_next:.2f}%, **{outcome}** the benchmark.",
                    icon="📋")
        st.markdown("**All models**")
        st.dataframe(pd.DataFrame({"Model": MODEL_ORDER,
                                   "P(above)": [f"{probs[m]:.1%}" for m in MODEL_ORDER],
                                   "Threshold": [f"{meta['thresholds'][m]:.2f}" for m in MODEL_ORDER]}),
                     hide_index=True, width="stretch")
    with right:
        st.markdown("#### Why: the EBM's feature contributions")
        st.plotly_chart(charts.contributions(contrib), width="stretch", theme=None)
        st.markdown(explain_text(contrib))
        st.caption("Blue bars raise the chance of beating the benchmark; red bars lower it.")
        st.caption(f"EBM is additive: intercept ({intercept:+.2f}) + the sum of all {len(contrib)} contributions "
                   f"= log-odds {intercept + contrib['contribution'].sum():+.2f}, i.e. probability {p:.1%}. "
                   "Every contribution is read directly off the model, not estimated after the fact.")
    with st.expander("Computed model inputs"):
        st.dataframe(X.T.rename(columns={X.index[0]: "value"}).assign(
            feature=lambda d: d.index.map(config.FEATURE_LABELS))[["feature", "value"]].astype(str), width="stretch")


# ---------------------------------------------------------------- watchlist

def render_watchlist():
    st.markdown("Score every active campaign and rank them by the chance of beating the benchmark tomorrow, "
                "so the ones at risk surface first.")
    src = st.radio("Data", ["Bundled dataset", "Upload a CSV"], horizontal=True, key="watch_src")
    full = panel
    if src == "Bundled dataset":
        dates = sorted(panel["date"].dt.date.unique())
        default = next((d for d in dates if str(d) > meta["val_end"]), dates[-1])
        as_of = st.select_slider("Score as of", options=dates, value=default)
        sub = panel[panel["date"].dt.date <= as_of]
        active = sub.loc[sub["date"].dt.date == as_of, "campaign_id"]
        sub = sub[sub["campaign_id"].isin(active)]
    else:
        st.caption("CSV with one row per campaign per day: date, campaign_id, platform, objective, spend, "
                   "impressions, clicks. Raw Google Ads / Meta Ads exports can be converted with "
                   "`python -m src.data`.")
        up = st.file_uploader("Campaign-day CSV", type="csv")
        if up is None:
            return
        raw = data.standardise_columns(pd.read_csv(up))
        missing = [c for c in data.REQUIRED_COLUMNS if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            return
        sub = full = data.clean(raw[data.REQUIRED_COLUMNS], verbose=False)

    keys = bench_keys(sub)
    unknown = sorted(set(keys) - set(benchmark))
    if unknown:
        st.warning(f"Skipped campaigns with no benchmark (not in training data): {', '.join(unknown)}")
        sub = sub[~keys.isin(unknown)]

    feats = features.build_features(sub, benchmark, include_unlabelled=True)
    latest = feats.sort_values("date").groupby("campaign_id").tail(1).reset_index(drop=True)
    if latest.empty:
        st.info("No campaigns to score.")
        return
    Xw = latest[config.FEATURES]
    latest["p"] = ebm.predict_proba(Xw)[:, 1]
    terms = ebm.eval_terms(Xw)
    latest["main_drag"] = [explain.term_label(ebm.term_names_[i]) for i in terms.argmin(axis=1)]
    nxt = (full.assign(date=full["date"] - pd.Timedelta(days=1))[["campaign_id", "date", "clicks", "impressions"]]
               .rename(columns={"clicks": "clicks_next", "impressions": "impressions_next"}))
    latest = latest.merge(nxt, on=["campaign_id", "date"], how="left")
    latest["actual_ctr"] = np.where(latest["impressions_next"] > 0,
                                    100 * latest["clicks_next"] / latest["impressions_next"], np.nan)

    at_risk = latest["p"] < threshold
    m1, m2, m3 = st.columns(3)
    m1.metric("Campaigns scored", len(latest))
    m2.metric("At risk tomorrow", int(at_risk.sum()))
    m3.metric("On track", int((~at_risk).sum()))

    table = pd.DataFrame({
        "Status": np.where(at_risk, "⚠️ At risk", "✅ On track"),
        "Campaign": latest["campaign_id"],
        "Platform": latest["platform"],
        "Objective": latest["objective"],
        "Last day": latest["date"].dt.date,
        "CTR, 7 days (%)": latest["ctr_7d"].round(2),
        "Benchmark (%)": latest["benchmark_ctr"].round(2),
        "P(above benchmark)": latest["p"],
        "Biggest negative factor": latest["main_drag"],
        "Actual next-day CTR (%)": latest["actual_ctr"].round(2),
    }).sort_values("P(above benchmark)").reset_index(drop=True)
    if table["Actual next-day CTR (%)"].isna().all():
        table = table.drop(columns="Actual next-day CTR (%)")
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "P(above benchmark)": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    st.download_button("Download watchlist (CSV)", table.to_csv(index=False), "campaign_watchlist.csv", "text/csv")


# ---------------------------------------------------------------- what the model learned

def render_learned():
    imp = explain.ebm_importance(ebm)
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("#### Which features matter most")
        st.plotly_chart(charts.importance(imp), width="stretch", theme=None)
        st.caption("Importance = average absolute contribution of each term to the log-odds over the training "
                   "data. Terms with × are pairwise interactions the EBM learned automatically.")
    with c2:
        st.markdown("#### How each feature moves the prediction")
        mains = imp.loc[imp["kind"] == "main effect", "term"].tolist()
        term = st.selectbox("Feature", mains, format_func=lambda t: config.FEATURE_LABELS.get(t, t))
        shape = explain.ebm_shape(ebm, term)
        x_range = None
        if "x_left" in shape:
            x_range = [float(dataset[term].quantile(0.01)), float(dataset[term].quantile(0.99))]
        st.plotly_chart(charts.shape(shape, config.FEATURE_LABELS.get(term, term), x_range),
                        width="stretch", theme=None)
        st.caption("The EBM's learned function for this feature (shaded band: uncertainty across bagged "
                   "models). Above zero raises the chance of beating the benchmark; below zero lowers it.")

    st.divider()
    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("#### Interactions")
        inter = imp.loc[imp["kind"] == "interaction", "term"].tolist()
        if inter:
            it = st.selectbox("Interaction", inter, format_func=explain.term_label)
            left, right_, scores = explain.ebm_interaction(ebm, it)
            a, b = it.split(" & ")
            st.plotly_chart(charts.interaction(left, right_, scores, config.FEATURE_LABELS[a],
                                               config.FEATURE_LABELS[b]), width="stretch", theme=None)
            st.caption("Extra effect of the two features together, on top of their individual effects. "
                       "Hover a cell to see its value ranges.")
    with c2:
        st.markdown("#### Partial dependence: do the models agree?")
        numeric = [t for t in mains if t in config.NUMERIC_FEATURES]
        feat = st.selectbox("Feature ", numeric, format_func=lambda t: config.FEATURE_LABELS.get(t, t))
        X_test = dataset.loc[dataset["split"] == "test", config.FEATURES].reset_index(drop=True)
        grid, curves = cached_pdp(models, X_test, feat)
        st.plotly_chart(charts.pdp(grid, curves, config.FEATURE_LABELS[feat],
                                   ref_line=1.0 if feat.startswith("ctr_index") else None),
                        width="stretch", theme=None)
        st.caption("Average predicted probability on the test set when this feature is set to each value "
                   "for every campaign-day.")


# ---------------------------------------------------------------- performance

def render_performance():
    comparison, test_preds, tuning, metrics = load_reports()
    test = comparison[comparison["split"] == "test"].drop(columns="split").set_index("model")
    test["roc_auc_95ci"] = [
        "{:.3f}–{:.3f}".format(*metrics[m]["test"]["roc_auc_ci"]) for m in test.index]
    st.markdown("#### Held-out test set")
    st.caption(f"Test period: target dates after {meta['val_end']} ({meta['n_rows']['test']} campaign-days, "
               f"{meta['positive_rate']['test']:.0%} above benchmark). Models were tuned on validation data and "
               "refit on train + validation before this single evaluation.")
    st.dataframe(test[["threshold", "precision", "recall", "f1", "specificity", "balanced_accuracy",
                       "roc_auc", "roc_auc_95ci", "pr_auc", "brier"]]
                 .rename(columns={"specificity": "at-risk recall", "balanced_accuracy": "balanced acc.",
                                  "roc_auc_95ci": "ROC-AUC 95% CI"}),
                 width="stretch", column_config={c: st.column_config.NumberColumn(format="%.3f") for c in
                                                ["precision", "recall", "f1", "at-risk recall", "balanced acc.",
                                                 "roc_auc", "pr_auc", "brier"]}
                 | {"threshold": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Precision/recall/F1 treat 'above benchmark' as the positive class; at-risk recall is the share "
               "of below-benchmark days correctly flagged. The persistence baseline predicts 'tomorrow = today'.")

    y_test = test_preds[config.TARGET].to_numpy()
    probs = {m: test_preds[f"p_{m}"].to_numpy() for m in MODEL_ORDER + ["Persistence baseline"]}
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### ROC curve")
        st.plotly_chart(charts.roc(y_test, probs), width="stretch", theme=None)
    with c2:
        st.markdown("#### Precision-recall curve")
        st.plotly_chart(charts.pr(y_test, probs), width="stretch", theme=None)

    st.divider()
    m = st.selectbox("Model details for", MODEL_ORDER)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"#### Confusion matrix (threshold {meta['thresholds'][m]:.2f})")
        pred = (probs[m] >= meta["thresholds"][m]).astype(int)
        st.plotly_chart(charts.confusion(y_test, pred), width="stretch", theme=None)
    with c2:
        st.markdown("#### Hyperparameter search")
        st.caption("Grid search on the training period, scored by validation ROC-AUC. Best configuration first.")
        st.dataframe(tuning[tuning["model"] == m].dropna(axis=1, how="all").drop(columns="model")
                     .sort_values("val_roc_auc", ascending=False)
                     .rename(columns=lambda col: col.removeprefix("param_")),
                     hide_index=True, width="stretch")


# ---------------------------------------------------------------- data overview

def render_data():
    log = meta["cleaning_log"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Campaign-days (modelling rows)", f"{len(dataset):,}")
    c2.metric("Campaigns", f"{dataset['campaign_id'].nunique()}")
    start, end = (pd.Timestamp(d) for d in meta["date_range"])
    c3.metric("Days covered", f"{(end - start).days + 1}", help=f"{start:%d %b %Y} to {end:%d %b %Y}")
    c4.metric("Above benchmark", f"{dataset[config.TARGET].mean():.0%}")
    st.markdown("#### Weekly median CTR index by platform")
    st.caption("CTR index = daily CTR ÷ benchmark for the campaign's platform and objective (1 = at benchmark). "
               "Vertical lines mark where the validation and test periods start.")
    st.plotly_chart(charts.ctr_index_over_time(dataset, {"validation": meta["train_end"], "test": meta["val_end"]}),
                    width="stretch", theme=None)
    st.markdown("#### Daily CTR by objective and platform")
    st.plotly_chart(charts.ctr_by_group(dataset), width="stretch", theme=None)
    with st.expander("Rows per split and cleaning log"):
        st.dataframe(pd.DataFrame({"rows": meta["n_rows"], "share above benchmark": meta["positive_rate"]}),
                     width="stretch")
        st.json(log)


for tab, render in [(tab_predict, render_predict), (tab_watch, render_watchlist), (tab_learned, render_learned),
                    (tab_perf, render_performance), (tab_data, render_data)]:
    with tab:
        render()

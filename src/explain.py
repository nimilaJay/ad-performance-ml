"""Explainability analysis.

* EBM global explanations: term importances and learned shape functions (exact, not approximated).
* EBM local explanations: per-prediction additive contributions for individual campaign-days.
* Partial Dependence Plots comparing EBM, Logistic Regression and CatBoost.
* Permutation importance on the test set (model-agnostic, comparable across all three models).

Usage:  python -m src.explain   (run after src.train)
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.inspection import permutation_importance

from src import config
from src.models import MODEL_NAMES
from src.plotting import (AXIS, INK_SECONDARY, MODEL_COLORS, NEGATIVE, NEUTRAL, POSITIVE, SERIES, SURFACE,
                          save)


DAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def term_label(term: str) -> str:
    return " × ".join(config.FEATURE_LABELS.get(t, t) for t in term.split(" & "))


def load_artifacts():
    with open(config.MODELS_DIR / "metadata.json") as f:
        meta = json.load(f)
    loaded = {MODEL_NAMES[k]: joblib.load(config.MODELS_DIR / f"{k}.joblib") for k in MODEL_NAMES}
    df = pd.read_csv(config.DATA_PROCESSED / "model_dataset.csv", parse_dates=["date", "target_date"])
    return loaded, meta, df


# ---------------------------------------------------------------- EBM global and local explanations

def ebm_importance(ebm) -> pd.DataFrame:
    """Mean absolute contribution (log-odds) of each term over the training data."""
    return (pd.DataFrame({"term": ebm.term_names_, "importance": ebm.term_importances()})
              .assign(label=lambda d: d["term"].map(term_label),
                      kind=lambda d: np.where(d["term"].str.contains(" & "), "interaction", "main effect"))
              .sort_values("importance", ascending=False, ignore_index=True))


def ebm_shape(ebm, term: str) -> pd.DataFrame:
    """Learned shape function of a main-effect term: contribution (log-odds) per bin or category."""
    d = ebm.explain_global().data(ebm.term_names_.index(term))
    if d["type"] != "univariate":
        raise ValueError(f"{term} is not a main effect")
    names = list(d["names"])
    if len(names) == len(d["scores"]) + 1:  # continuous: names are bin edges
        return pd.DataFrame({"x_left": names[:-1], "x_right": names[1:], "score": d["scores"],
                             "lower": d["lower_bounds"], "upper": d["upper_bounds"]})
    out = pd.DataFrame({"category": names, "score": d["scores"],
                        "lower": d["lower_bounds"], "upper": d["upper_bounds"]})
    if term == "day_of_week":
        out = out.set_index("category").reindex([d for d in DAY_ORDER if d in names]).reset_index()
    return out


def ebm_interaction(ebm, term: str):
    d = ebm.explain_global().data(ebm.term_names_.index(term))
    return np.asarray(d["left_names"]), np.asarray(d["right_names"]), np.asarray(d["scores"])


def ebm_local(ebm, x_row: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Exact additive breakdown of one prediction: intercept + sum(contributions) = log-odds."""
    contrib = ebm.eval_terms(x_row[config.FEATURES])[0]
    row = x_row.iloc[0]
    values = [", ".join(_fmt_value(t, row[t]) for t in term.split(" & ")) for term in ebm.term_names_]
    out = (pd.DataFrame({"term": ebm.term_names_, "label": [term_label(t) for t in ebm.term_names_],
                         "value": values, "contribution": contrib})
             .sort_values("contribution", key=np.abs, ascending=False, ignore_index=True))
    return out, float(ebm.intercept_[0])


def _fmt_value(feature, v):
    if isinstance(v, str):
        return v
    if feature.startswith("ctr_index") or feature == "ctr_trend":
        return f"{v:.2f}"
    if feature.startswith("spend") or feature == "cpm_7d":
        return f"{v:,.2f}"
    return f"{v:,.0f}"


def partial_dependence(model, X: pd.DataFrame, feature: str, grid: np.ndarray) -> np.ndarray:
    """Average predicted probability when `feature` is set to each grid value for every row."""
    Xc = X[config.FEATURES].copy()
    out = []
    for v in grid:
        Xc[feature] = v
        out.append(model.predict_proba(Xc)[:, 1].mean())
    return np.asarray(out)


def feature_grid(X: pd.DataFrame, feature: str, n: int = 30) -> np.ndarray:
    return np.unique(np.quantile(X[feature].dropna(), np.linspace(0.02, 0.98, n)))


# ---------------------------------------------------------------- figures

def plot_ebm_importance(imp: pd.DataFrame, path, top=15):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.2))
    ax.barh(d["label"], d["importance"], height=0.6, color=SERIES[0])
    for y, v in enumerate(d["importance"]):
        ax.text(v, y, f" {v:.3f}", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set(xlabel="Mean |contribution| to log-odds", title=f"EBM term importance (top {top})")
    ax.grid(axis="y", visible=False)
    save(fig, path)


def plot_ebm_shapes(ebm, terms, path):
    n = len(terms)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.1 * rows))
    for ax, term in zip(axes.flat, terms):
        s = ebm_shape(ebm, term)
        ax.axhline(0, color=AXIS, lw=1)
        if "category" in s:
            colors = [POSITIVE if v >= 0 else NEGATIVE for v in s["score"]]
            ax.bar(s["category"].astype(str), s["score"], width=0.6, color=colors)
            ax.errorbar(range(len(s)), s["score"], yerr=[s["score"] - s["lower"], s["upper"] - s["score"]],
                        fmt="none", ecolor=INK_SECONDARY, elinewidth=1, capsize=0)
            ax.tick_params(axis="x", labelrotation=30 if len(s) > 4 else 0)
            ax.grid(axis="x", visible=False)
        else:
            # Clip the open-ended outer bins to the observed range for display.
            x = np.r_[s["x_left"].to_numpy(), s["x_right"].iloc[-1]]
            y = np.r_[s["score"].to_numpy(), s["score"].iloc[-1]]
            lo = np.r_[s["lower"].to_numpy(), s["lower"].iloc[-1]]
            hi = np.r_[s["upper"].to_numpy(), s["upper"].iloc[-1]]
            ax.fill_between(x, lo, hi, step="post", color=SERIES[0], alpha=0.12, lw=0)
            ax.step(x, y, where="post", color=SERIES[0])
            if term.startswith("ctr_index"):
                ax.axvline(1.0, color=AXIS, lw=1)
                ax.set_xlim(0, min(x.max(), 3))
            elif x.max() > 50 * max(np.median(x), 1):
                ax.set_xscale("symlog")
        ax.set_title(config.FEATURE_LABELS.get(term, term), fontsize=10)
        ax.set_ylabel("Contribution (log-odds)" if ax in axes[:, 0] else "")
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle("EBM shape functions: how each feature moves the log-odds of beating the benchmark",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def plot_interaction(ebm, term, path):
    left, right, scores = ebm_interaction(ebm, term)
    a, b = term.split(" & ")
    lim = np.abs(scores).max()
    cmap = LinearSegmentedColormap.from_list("div", [NEGATIVE, NEUTRAL, POSITIVE])
    fig, ax = plt.subplots(figsize=(6, 4.6))
    # Bin edges are numeric; plot on bin index so open-ended edge bins stay readable.
    mesh = ax.pcolormesh(scores.T, cmap=cmap, vmin=-lim, vmax=lim)
    for edges, set_ticks in [(left, ax.set_xticks), (right, ax.set_yticks)]:
        idx = np.linspace(0, len(edges) - 1, 6).astype(int)
        set_ticks(idx, [f"{edges[i]:.3g}" for i in idx])
    ax.set(xlabel=config.FEATURE_LABELS.get(a, a), ylabel=config.FEATURE_LABELS.get(b, b),
           title=f"EBM interaction: {term_label(term)}")
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label="Contribution (log-odds)")
    save(fig, path)


def plot_pdp(models_by_name, X, features, path):
    cols = 2
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.4 * rows), sharey=True)
    for ax, feat in zip(axes.flat, features):
        grid = feature_grid(X, feat)
        for name, model in models_by_name.items():
            ax.plot(grid, partial_dependence(model, X, feat, grid), color=MODEL_COLORS[name], label=name)
        if feat.startswith("ctr_index"):
            ax.axvline(1.0, color=AXIS, lw=1)
        ax.set_title(config.FEATURE_LABELS.get(feat, feat), fontsize=10)
        ax.set_ylim(0, 1)
    for ax in axes[:, 0]:
        ax.set_ylabel("Avg P(above benchmark)")
    for ax in axes.flat[len(features):]:
        ax.set_visible(False)
    axes.flat[0].legend(loc="upper left")
    fig.suptitle("Partial dependence, test set", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def plot_permutation(perm: pd.DataFrame, path):
    order = perm.groupby("feature")["importance"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(7, 0.36 * len(order) + 1.2))
    y = {f: i for i, f in enumerate(order)}
    offsets = {name: off for name, off in zip(MODEL_COLORS, (-0.2, 0, 0.2))}
    for name, g in sorted(perm.groupby("model"), key=lambda kv: list(MODEL_COLORS).index(kv[0])):
        ys = g["feature"].map(y) + offsets[name]
        ax.errorbar(g["importance"], ys, xerr=g["std"], fmt="none", ecolor=MODEL_COLORS[name], elinewidth=1)
        ax.scatter(g["importance"], ys, s=36, color=MODEL_COLORS[name], edgecolor=SURFACE, linewidth=1.5,
                   label=name, zorder=3)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_yticks(range(len(order)), [config.FEATURE_LABELS[f] for f in order])
    ax.set(xlabel="Drop in test ROC-AUC when the feature is shuffled",
           title="Permutation importance, test set")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save(fig, path)


def plot_local(contrib: pd.DataFrame, intercept, prob, title, path, top=10):
    d = contrib.head(top).iloc[::-1]
    rest = contrib["contribution"].iloc[top:].sum()
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * top + 1.6))
    colors = [POSITIVE if v >= 0 else NEGATIVE for v in d["contribution"]]
    labels = [f"{lab} = {val}" for lab, val in zip(d["label"], d["value"])]
    ax.barh(labels, d["contribution"], height=0.6, color=colors)
    ax.axvline(0, color=AXIS, lw=1)
    ax.set(xlabel="Contribution to log-odds (blue raises, red lowers P(above benchmark))",
           title=f"{title}\nP(above benchmark) = {prob:.1%}  ·  intercept {intercept:+.2f}  ·  "
                 f"{len(contrib) - top} other terms {rest:+.2f}")
    ax.grid(axis="y", visible=False)
    save(fig, path)


# ---------------------------------------------------------------- main

def main():
    loaded, meta, df = load_artifacts()
    ebm = loaded["EBM"]
    test = df[df["split"] == "test"].reset_index(drop=True)
    X_test, y_test = test[config.FEATURES], test[config.TARGET].to_numpy()
    fig_dir = config.FIGURES_DIR
    print("Saving explainability figures")

    imp = ebm_importance(ebm)
    imp.to_csv(config.REPORTS_DIR / "ebm_term_importance.csv", index=False)
    plot_ebm_importance(imp, fig_dir / "ebm_global_importance.png")

    main_terms = imp.loc[imp["kind"] == "main effect", "term"].head(9).tolist()
    plot_ebm_shapes(ebm, main_terms, fig_dir / "ebm_shape_functions.png")

    top_interaction = imp.loc[imp["kind"] == "interaction", "term"].head(1).tolist()
    if top_interaction:
        plot_interaction(ebm, top_interaction[0], fig_dir / "ebm_top_interaction.png")

    numeric_top = [t for t in main_terms if t in config.NUMERIC_FEATURES][:4]
    plot_pdp(loaded, X_test, numeric_top, fig_dir / "pdp_model_comparison.png")

    rows = []
    for name, model in loaded.items():
        r = permutation_importance(model, X_test, y_test, scoring="roc_auc", n_repeats=10,
                                   random_state=config.RANDOM_STATE, n_jobs=-1)
        rows += [{"model": name, "feature": f, "importance": m, "std": s}
                 for f, m, s in zip(config.FEATURES, r.importances_mean, r.importances_std)]
    perm = pd.DataFrame(rows)
    perm.to_csv(config.REPORTS_DIR / "permutation_importance.csv", index=False)
    plot_permutation(perm, fig_dir / "permutation_importance.png")

    # Local explanations for three representative test cases.
    p = ebm.predict_proba(X_test)[:, 1]
    thr = meta["thresholds"]["EBM"]
    pred = (p >= thr).astype(int)
    cases = {
        "local_confident_above": np.where((pred == 1) & (y_test == 1))[0][np.argmax(p[(pred == 1) & (y_test == 1)])],
        "local_confident_below": np.where((pred == 0) & (y_test == 0))[0][np.argmin(p[(pred == 0) & (y_test == 0)])],
        "local_misclassified": np.where(pred != y_test)[0][np.argmax(np.abs(p - thr)[pred != y_test])],
    }
    case_rows = []
    for fname, i in cases.items():
        row = test.iloc[[i]]
        contrib, intercept = ebm_local(ebm, row)
        actual = "above" if y_test[i] else "below"
        title = (f"{row['campaign_id'].iat[0]} · {row['platform'].iat[0]} {row['objective'].iat[0]} · "
                 f"{row['target_date'].iat[0].date()} (actual: {actual} benchmark, "
                 f"CTR {row['next_ctr'].iat[0]:.3f}% vs {row['benchmark_ctr'].iat[0]:.3f}%)")
        plot_local(contrib, intercept, p[i], title, fig_dir / f"{fname}.png")
        case_rows.append({"case": fname, "campaign_id": row["campaign_id"].iat[0],
                          "target_date": str(row["target_date"].iat[0].date()), "probability": p[i],
                          "actual": actual, "top_contributions": contrib.head(5)[["label", "value", "contribution"]]
                          .round(3).to_dict("records")})

    summary = {
        "top_terms": imp.head(10)[["label", "kind", "importance"]].round(4).to_dict("records"),
        "shape_summaries": {t: ebm_shape(ebm, t).round(3).to_dict("list") for t in
                            ["platform", "objective", "day_of_week"]},
        "permutation_importance_top5": {m: g.nlargest(5, "importance")[["feature", "importance"]]
                                        .round(4).to_dict("records") for m, g in perm.groupby("model")},
        "local_cases": case_rows,
    }
    with open(config.REPORTS_DIR / "explainability_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print("\nTop EBM terms:")
    print(imp.head(10)[["label", "kind", "importance"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Evaluation metrics, bootstrap confidence intervals and evaluation plots."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)

from src import config
from src.plotting import AXIS, INK, MODEL_COLORS, SEQUENTIAL, save


def classification_metrics(y, p, threshold=0.5) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    pred = (p >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        # Share of below-benchmark days the model flags: the "at-risk campaign" catch rate.
        "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        # Brier score only applies to probabilities (not to the persistence baseline's raw score).
        "brier": brier_score_loss(y, p) if 0 <= p.min() and p.max() <= 1 else np.nan,
    }


def best_threshold(y, p) -> float:
    """Decision threshold maximising Youden's J (sensitivity + specificity - 1) on validation data.

    Maximising positive-class F1 instead pushes the threshold down and rarely flags under-performing
    campaigns, which is the main use of the dashboard, so both error types are weighted equally.
    """
    fpr, tpr, thresholds = roc_curve(y, p)
    return float(np.clip(thresholds[int(np.argmax(tpr - fpr))], 0.0, 1.0))


def bootstrap_ci(y, p, threshold, groups, n_boot=1000, seed=0) -> dict:
    """95% percentile bootstrap intervals for ROC-AUC and F1 on the test set.

    Days from the same campaign are correlated, so whole campaigns are resampled (cluster bootstrap);
    resampling individual days would treat them as independent and understate the uncertainty.
    """
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    rows_by_group = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(seed)
    aucs, f1s = [], []
    for _ in range(n_boot):
        picked = rng.integers(0, len(rows_by_group), len(rows_by_group))
        idx = np.concatenate([rows_by_group[i] for i in picked])
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        f1s.append(f1_score(y[idx], (p[idx] >= threshold).astype(int), zero_division=0))
    return {"roc_auc_ci": [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))],
            "f1_ci": [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))]}


def segment_metrics(df, prob_col, threshold, by) -> pd.DataFrame:
    """Test performance per segment (e.g. platform or objective), to check no group is served badly."""
    rows = []
    for seg, g in df.groupby(by):
        y, p = g[config.TARGET].to_numpy(), g[prob_col].to_numpy()
        pred = (p >= threshold).astype(int)
        rows.append({
            "segment_type": by, "segment": seg, "n": len(g), "share_above": y.mean(),
            "roc_auc": roc_auc_score(y, p) if 0 < y.mean() < 1 else np.nan,
            "f1": f1_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "specificity": recall_score(y, pred, pos_label=0, zero_division=0),
        })
    return pd.DataFrame(rows)


def plot_roc(y, probs: dict, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, label="Chance (AUC 0.50)")
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        ax.plot(fpr, tpr, color=MODEL_COLORS[name], label=f"{name} (AUC {roc_auc_score(y, p):.3f})",
                lw=1.5 if name == "Persistence baseline" else 2)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01),
           title="ROC curves, test set")
    ax.legend(loc="lower right")
    save(fig, path)


def plot_pr(y, probs: dict, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    base = float(np.mean(y))
    ax.axhline(base, color=AXIS, lw=1, label=f"Base rate ({base:.2f})")
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y, p)
        ax.plot(rec, prec, color=MODEL_COLORS[name], label=f"{name} (AP {average_precision_score(y, p):.3f})",
                lw=1.5 if name == "Persistence baseline" else 2)
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.01),
           title="Precision-recall curves, test set")
    ax.legend(loc="lower left")
    save(fig, path)


def plot_confusion(y, preds: dict, path):
    cmap = LinearSegmentedColormap.from_list("seq", SEQUENTIAL)
    fig, axes = plt.subplots(1, len(preds), figsize=(3.4 * len(preds), 3.3))
    labels = ["Below", "Above"]
    for ax, (name, pred) in zip(np.atleast_1d(axes), preds.items()):
        cm = confusion_matrix(y, pred, labels=[0, 1])
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.sum())
        for i in range(2):
            for j in range(2):
                share = cm[i, j] / cm.sum()
                ax.text(j, i, f"{cm[i, j]}\n{share:.0%}", ha="center", va="center", fontsize=10, color=INK)
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=labels, yticklabels=labels,
               xlabel="Predicted", ylabel="Actual", title=name)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("Confusion matrices, test set (next-day CTR vs benchmark)", x=0.02, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    save(fig, path)


def plot_calibration(y, probs: dict, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=AXIS, lw=1, label="Perfectly calibrated")
    for name, p in probs.items():
        if name == "Persistence baseline":
            continue
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, color=MODEL_COLORS[name], marker="o", ms=5,
                markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=name)
    ax.set(xlabel="Mean predicted probability", ylabel="Observed share above benchmark",
           xlim=(0, 1), ylim=(0, 1), title="Calibration, test set")
    ax.legend(loc="upper left")
    save(fig, path)

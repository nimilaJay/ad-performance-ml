"""Train, tune and evaluate EBM, Logistic Regression and CatBoost.

Steps:
  1. Load and clean data, engineer features, split chronologically (train / validation / test).
  2. Grid-search each model on train, scored by validation ROC-AUC.
  3. Choose each model's decision threshold on validation (max Youden's J).
  4. Refit the best configuration on train + validation, then evaluate once on the held-out test set.

Usage:  python -m src.train [--data path/to/campaign_daily.csv]
"""

import argparse
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, data, evaluate, features, models


def tune(model_key, X_tr, y_tr, X_val, y_val):
    rows = []
    for params in models.grid(model_key):
        t0 = time.time()
        model = models.fit(model_key, models.make_model(model_key, params), X_tr, y_tr, X_val, y_val)
        p_val = model.predict_proba(X_val)[:, 1]
        row = {"model": models.MODEL_NAMES[model_key], **{f"param_{k}": v for k, v in params.items()},
               "val_roc_auc": roc_auc_score(y_val, p_val), "fit_seconds": round(time.time() - t0, 1)}
        if model_key == "catboost":
            row["best_iteration"] = model.get_best_iteration()
        rows.append((row, params, model, p_val))
        print(f"  {params} -> val AUC {row['val_roc_auc']:.4f}")
    best_row, best_params, best_model, best_p_val = max(rows, key=lambda r: r[0]["val_roc_auc"])
    return [r[0] for r in rows], best_row, best_params, best_model, best_p_val


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=None, help="Standard-schema CSV (default: real data if present, else synthetic)")
    args = parser.parse_args()

    panel = data.load(args.data)
    feats, meta = features.prepare_dataset(panel)
    feats.to_csv(config.DATA_PROCESSED / "model_dataset.csv", index=False)
    print(f"Dataset: {len(feats):,} campaign-days | benchmark CTR (%): {meta['benchmark_ctr_pct']}")

    parts = {s: feats[feats["split"] == s] for s in ["train", "val", "test"]}
    X = {s: d[config.FEATURES] for s, d in parts.items()}
    y = {s: d[config.TARGET].to_numpy() for s, d in parts.items()}
    for s in parts:
        print(f"  {s:5s}: {len(parts[s]):5,} rows, {y[s].mean():.1%} above benchmark, "
              f"{parts[s]['target_date'].min().date()} to {parts[s]['target_date'].max().date()}")

    test_groups = parts["test"]["campaign_id"].to_numpy()
    X_trval = pd.concat([X["train"], X["val"]])
    y_trval = np.concatenate([y["train"], y["val"]])

    tuning_rows, results, test_probs, final_models = [], {}, {}, {}
    for key in ["ebm", "logreg", "catboost"]:
        name = models.MODEL_NAMES[key]
        print(f"\nTuning {name}")
        rows, best_row, best_params, tuned_model, p_val = tune(key, X["train"], y["train"], X["val"], y["val"])
        tuning_rows += rows
        threshold = evaluate.best_threshold(y["val"], p_val)

        final_params = dict(best_params)
        if key == "catboost":
            final_params.update(iterations=best_row["best_iteration"] + 1, early_stopping_rounds=None)
        final = models.fit(key, models.make_model(key, final_params), X_trval, y_trval)
        p_test = final.predict_proba(X["test"])[:, 1]

        results[name] = {
            "best_params": best_params,
            "validation": evaluate.classification_metrics(y["val"], p_val, threshold),
            "test": {**evaluate.classification_metrics(y["test"], p_test, threshold),
                     **evaluate.bootstrap_ci(y["test"], p_test, threshold, test_groups)},
        }
        test_probs[name] = p_test
        final_models[key] = final
        joblib.dump(final, config.MODELS_DIR / f"{key}.joblib")
        print(f"  best {best_params} | threshold {threshold:.2f} | "
              f"test AUC {results[name]['test']['roc_auc']:.4f}, F1 {results[name]['test']['f1']:.4f}")

    # Naive persistence baseline: "tomorrow will be like today". Its score is the last-day CTR index.
    persist = {s: parts[s]["ctr_index_1d"].fillna(parts[s]["ctr_index_7d"]).to_numpy() for s in ["val", "test"]}
    results["Persistence baseline"] = {
        "best_params": {"rule": "predict above if last-day CTR index > 1"},
        "validation": evaluate.classification_metrics(y["val"], persist["val"], 1.0),
        "test": {**evaluate.classification_metrics(y["test"], persist["test"], 1.0),
                 **evaluate.bootstrap_ci(y["test"], persist["test"], 1.0, test_groups)},
    }
    test_probs["Persistence baseline"] = persist["test"]

    # Save tables
    pd.DataFrame(tuning_rows).to_csv(config.REPORTS_DIR / "tuning_results.csv", index=False)
    comparison = pd.DataFrame([{"model": m, "split": s, **{k: v for k, v in r[s].items() if not k.endswith("_ci")}}
                               for m, r in results.items() for s in ["validation", "test"]])
    comparison.round(4).to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    with open(config.REPORTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2, default=float)

    preds = parts["test"][["campaign_id", "target_date", "platform", "objective", "ctr_index_1d",
                            "benchmark_ctr", "next_ctr", config.TARGET]].copy()
    for name, p in test_probs.items():
        preds[f"p_{name}"] = p
    preds.to_csv(config.REPORTS_DIR / "test_predictions.csv", index=False)
    segments = pd.concat([evaluate.segment_metrics(preds, f"p_{m}", results[m]["validation"]["threshold"], by)
                          .assign(model=m) for m in test_probs if m != "Persistence baseline"
                          for by in ["platform", "objective"]])
    segments.round(4).to_csv(config.REPORTS_DIR / "segment_performance.csv", index=False)

    metadata = {
        **meta,
        "data_path": panel.attrs["source_path"],
        "is_synthetic": panel.attrs["is_synthetic"],
        "cleaning_log": panel.attrs["cleaning_log"],
        "n_rows": {s: int(len(d)) for s, d in parts.items()},
        "positive_rate": {s: float(y[s].mean()) for s in parts},
        "features": config.FEATURES,
        "categorical_features": config.CATEGORICAL_FEATURES,
        "numeric_features": config.NUMERIC_FEATURES,
        "thresholds": {m: r["validation"]["threshold"] for m, r in results.items()},
        "best_params": {m: r["best_params"] for m, r in results.items()},
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(config.MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    # Plots
    print("\nSaving evaluation figures")
    fig_dir = config.FIGURES_DIR
    evaluate.plot_roc(y["test"], test_probs, fig_dir / "roc_curves.png")
    evaluate.plot_pr(y["test"], test_probs, fig_dir / "pr_curves.png")
    evaluate.plot_calibration(y["test"], test_probs, fig_dir / "calibration.png")
    evaluate.plot_confusion(
        y["test"],
        {m: (p >= results[m]["validation"]["threshold"]).astype(int) for m, p in test_probs.items()
         if m != "Persistence baseline"},
        fig_dir / "confusion_matrices.png")

    print("\nTest-set comparison")
    print(comparison[comparison["split"] == "test"].drop(columns="split").round(3).to_string(index=False))


if __name__ == "__main__":
    main()

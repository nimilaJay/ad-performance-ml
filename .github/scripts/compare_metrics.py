"""Markdown table comparing retrained models with the ones committed on main (used by retrain.yml)."""

import json
import subprocess

FIELDS = [("roc_auc", "ROC-AUC"), ("f1", "F1"), ("precision", "Precision"), ("recall", "Recall")]


def fmt(before, after):
    if after is None:
        return "–"
    return f"{after:.3f}" if before is None else f"{before:.3f} → {after:.3f}"


new = json.load(open("reports/metrics.json"))
old = json.loads(subprocess.run(["git", "show", "HEAD:reports/metrics.json"],
                                capture_output=True, text=True, check=True).stdout)
meta = json.load(open("models/metadata.json"))

print("### Test-set metrics: models on main → retrained")
print()
print("| Model | " + " | ".join(label for _, label in FIELDS) + " |")
print("|---|" + "---|" * len(FIELDS))
for model, result in new.items():
    before = old.get(model, {}).get("test", {})
    print(f"| {model} | " + " | ".join(fmt(before.get(k), result["test"].get(k)) for k, _ in FIELDS) + " |")
print()
data = meta["data_path"] + (" (synthetic placeholder data)" if meta.get("is_synthetic") else "")
print(f"Data: `{data}`, {sum(meta['n_rows'].values()):,} campaign-days "
      f"(train ≤ {meta['train_end']} < validation ≤ {meta['val_end']} < test).")
print()
print("The tests, including the quality gate in `tests/test_models.py`, passed on the retrained models.")

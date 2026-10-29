"""Saved models: they load, reproduce the reported results, explain exactly, and pass the quality gate."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import config

MODELS = ["EBM", "Logistic Regression", "CatBoost"]

# Deployment gate: a release is blocked unless the EBM reaches this test ROC-AUC and beats the
# naive "tomorrow = today" rule.
MIN_TEST_ROC_AUC = 0.80


@pytest.fixture(scope="module")
def test_rows(dataset):
    return dataset[dataset["split"] == "test"].reset_index(drop=True)


@pytest.fixture(scope="module")
def metrics():
    with open(config.REPORTS_DIR / "metrics.json") as f:
        return json.load(f)


@pytest.mark.parametrize("name", MODELS)
def test_model_reproduces_saved_test_predictions(models, test_rows, name):
    saved = pd.read_csv(config.REPORTS_DIR / "test_predictions.csv", parse_dates=["target_date"])
    merged = test_rows.merge(saved[["campaign_id", "target_date", f"p_{name}"]],
                             on=["campaign_id", "target_date"], validate="one_to_one")
    assert len(merged) == len(test_rows) == len(saved)
    p = models[name].predict_proba(merged[config.FEATURES])[:, 1]
    np.testing.assert_allclose(p, merged[f"p_{name}"], atol=1e-6)


def test_ebm_explanation_adds_up_to_the_prediction(models, test_rows):
    ebm = models["EBM"]
    X = test_rows[config.FEATURES]
    log_odds = ebm.intercept_[0] + ebm.eval_terms(X).sum(axis=1)
    np.testing.assert_allclose(1 / (1 + np.exp(-log_odds)), ebm.predict_proba(X)[:, 1], atol=1e-9)


@pytest.mark.parametrize("name", MODELS)
def test_thresholds_come_from_validation(meta, metrics, name):
    assert 0 < meta["thresholds"][name] < 1
    assert meta["thresholds"][name] == pytest.approx(metrics[name]["validation"]["threshold"])


def test_quality_gate(models, test_rows, metrics):
    auc = roc_auc_score(test_rows[config.TARGET], models["EBM"].predict_proba(test_rows[config.FEATURES])[:, 1])
    assert auc == pytest.approx(metrics["EBM"]["test"]["roc_auc"])
    assert auc >= MIN_TEST_ROC_AUC
    assert auc > metrics["Persistence baseline"]["test"]["roc_auc"]

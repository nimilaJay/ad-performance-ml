"""Model definitions and hyperparameter search spaces.

* EBM (Explainable Boosting Machine) - main model, a glass-box generalised additive model with
  pairwise interactions, trained by cyclic gradient boosting.
* Logistic Regression - linear, interpretable baseline.
* CatBoost - high-accuracy black-box gradient-boosted trees, as an upper-bound reference.
"""

from itertools import product

import numpy as np
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src import config

# Heavy-tailed volume features are log-transformed for the linear model only; EBM and CatBoost
# bin or split features, so monotone transforms do not change them.
LOG_FEATURES = ["campaign_day", "spend_1d", "spend_7d_avg", "spend_cum", "impressions_1d",
                "impressions_7d_avg", "clicks_1d", "clicks_7d_avg", "cpm_7d"]

MODEL_NAMES = {"ebm": "EBM", "logreg": "Logistic Regression", "catboost": "CatBoost"}

SEARCH_SPACES = {
    "ebm": {
        "interactions": [0, 5, 10],
        "max_bins": [32, 128],
        "min_samples_leaf": [4, 20],
        "learning_rate": [0.015],
    },
    "logreg": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": [None, "balanced"],
    },
    "catboost": {
        "depth": [4, 6],
        "learning_rate": [0.03, 0.1],
        "l2_leaf_reg": [3, 10],
    },
}


def grid(model_key: str) -> list[dict]:
    space = SEARCH_SPACES[model_key]
    return [dict(zip(space, values)) for values in product(*space.values())]


def make_model(model_key: str, params: dict):
    if model_key == "ebm":
        return ExplainableBoostingClassifier(
            feature_names=config.FEATURES,
            feature_types=["nominal"] * len(config.CATEGORICAL_FEATURES) + ["continuous"] * len(config.NUMERIC_FEATURES),
            random_state=config.RANDOM_STATE,
            **params,
        )

    if model_key == "logreg":
        log_cols = [c for c in config.NUMERIC_FEATURES if c in LOG_FEATURES]
        lin_cols = [c for c in config.NUMERIC_FEATURES if c not in LOG_FEATURES]
        numeric = lambda transform: Pipeline([  # noqa: E731
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("transform", transform),
            ("scale", StandardScaler()),
        ])
        pre = ColumnTransformer([
            ("cat", OneHotEncoder(handle_unknown="ignore"), config.CATEGORICAL_FEATURES),
            ("log", numeric(FunctionTransformer(np.log1p)), log_cols),
            ("lin", numeric("passthrough"), lin_cols),
        ])
        return Pipeline([("pre", pre),
                         ("clf", LogisticRegression(max_iter=5000, random_state=config.RANDOM_STATE, **params))])

    if model_key == "catboost":
        # Early stopping on validation picks the number of trees; the final refit (no eval set)
        # passes that number back in as `iterations` with early_stopping_rounds=None.
        defaults = {"iterations": 2000, "early_stopping_rounds": 100}
        return CatBoostClassifier(
            eval_metric="AUC",
            cat_features=config.CATEGORICAL_FEATURES,
            random_seed=config.RANDOM_STATE,
            verbose=0,
            allow_writing_files=False,
            **{**defaults, **params},
        )

    raise ValueError(f"Unknown model {model_key}")


def fit(model_key: str, model, X, y, X_val=None, y_val=None):
    """Fit a model; CatBoost uses the validation set for early stopping when it is given."""
    if model_key == "catboost" and X_val is not None:
        model.fit(X, y, eval_set=(X_val, y_val), use_best_model=True)
    else:
        model.fit(X, y)
    return model

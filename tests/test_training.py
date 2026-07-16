"""Smoke tests for training helpers (no full model fit)."""

import pandas as pd

from src.training import build_classifier, split_features_targets


def test_split_features_targets_drops_churn():
    frame = pd.DataFrame(
        {
            "tenure": [1, 12],
            "MonthlyCharges": [20.0, 50.0],
            "Churn": [0, 1],
        }
    )
    X, y = split_features_targets(frame)
    assert "Churn" not in X.columns
    assert y.tolist() == [0, 1]


def test_build_classifier_uses_configured_objective():
    model = build_classifier()
    assert model.objective == "binary:logistic"
    assert model.n_estimators > 0

import pandas as pd
import numpy as np
from src.features import add_feature_columns


def test_add_feature_columns_creates_expected_columns():
    df = pd.DataFrame(
        {
            "customerID": ["0001", "0002"],
            "tenure": [1, 12],
            "TotalCharges": [50.0, 600.0],
            "MonthlyCharges": [50.0, 50.0],
            "PhoneService": ["Yes", "No"],
            "InternetService": ["DSL", "No"],
            "OnlineSecurity": ["No", "Yes"],
            "Contract": ["Month-to-month", "Two year"],
        }
    )

    featured = add_feature_columns(df)

    assert "tenure_bucket" in featured.columns
    assert "total_service_count" in featured.columns
    assert "charge_to_tenure_ratio" in featured.columns
    assert "avg_monthly_charge" in featured.columns
    assert "contract_risk_score" in featured.columns

    # Check that counts are integers and ratios are finite
    assert featured["total_service_count"].dtype == int
    assert np.isfinite(featured["charge_to_tenure_ratio"].fillna(0)).all()

import pandas as pd
from src.preprocessing import clean_telco_data


def test_clean_telco_data_drops_zero_tenure_and_blank_totalcharges():
    raw = pd.DataFrame([
        {
            "customerID": "0001",
            "tenure": "0",
            "SeniorCitizen": "0",
            "TotalCharges": "",
            "MonthlyCharges": "0.0",
            "Partner": "Yes",
            "Dependents": "No",
            "PhoneService": "Yes",
            "PaperlessBilling": "No",
            "Churn": "No",
        },
        {
            "customerID": "0002",
            "tenure": "12",
            "SeniorCitizen": "0",
            "TotalCharges": "120.0",
            "MonthlyCharges": "10.0",
            "Partner": "No",
            "Dependents": "No",
            "PhoneService": "No",
            "PaperlessBilling": "Yes",
            "Churn": "Yes",
        },
    ])

    cleaned, report = clean_telco_data(raw, drop_zero_tenure_rows=True)

    # One row should be dropped (tenure 0 with blank TotalCharges)
    assert report.rows_before == 2
    assert report.rows_after == 1
    assert cleaned.shape[0] == 1
    # Ensure Churn was encoded to int
    assert cleaned['Churn'].dtype == int
    assert cleaned['Churn'].iloc[0] == 1

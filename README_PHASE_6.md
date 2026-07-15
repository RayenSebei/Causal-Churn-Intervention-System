# Causal Churn Intervention System

**This is not a churn classifier — it's a decision-support tool that answers: "Who should we spend retention budget on?"**

## Business Problem

Customer churn is costly, but retention budgets are finite. Indiscriminate discounting is wasteful. This system identifies which customers are most responsive to intervention (high uplift) vs. those who will churn anyway (lost causes) or won't churn without help (sure things), enabling allocation of limited retention budget to maximally impactful targets.

---

## Quick Start

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the dashboard
```bash
python dashboard/app.py
```
Then navigate to **http://127.0.0.1:8050** in your browser.

The dashboard will show:
- **ROI Comparison** (top): Targeted (Persuadables + Sleeping Dogs) vs. Blanket spend — the core business case.
- **Customer Risk Table**: Filtered by segment and contract type, with churn probability, uplift, and SHAP explanations.
- **Segment Breakdown**: Visual distribution of customers across four segments.
- **Customer Detail View**: Click any customer to see personalized churn drivers and treatment benefit.

---

## System Architecture

### Phases

#### Phase 1: Data & EDA
- **Module**: `src/data_prep.py`, `src/features.py`
- **Notebook**: `notebooks/eda.ipynb`
- Explicit handling of `TotalCharges` (coerced numeric, 11 rows with tenure==0 and blank charges dropped).
- Feature engineering: tenure buckets, total service count, charge-to-tenure ratio.
- EDA: churn-rate breakdowns by contract, tenure, monthly charges, service count.

#### Phase 2: Baseline Prediction
- **Module**: `src/model.py`
- XGBoost classifier with stratified 5-fold cross-validation.
- Class imbalance: SMOTE applied only inside training folds (not on validation/test data).
- **Metrics**: ROC-AUC 0.8451 (CV), 0.8361 (test); Precision 0.578, Recall 0.612.
- Calibration curve visualized to ensure probability estimates are trustworthy.

#### Phase 3: Explainability
- **Module**: `src/explain.py`
- SHAP TreeExplainer for the XGBoost model.
- Per-customer narrative explanations: "Churn risk driven by [top 3 SHAP features with direction]."
- Example: "Month-to-month contract (↑ churn 0.629), Low tenure (↑ churn 0.301), No online security (↑ churn 0.259)."

#### Phase 4: Causal / Uplift Modeling
- **Module**: `src/uplift.py`
- **Synthetic treatment assignment**: Discount offered with higher probability to high-charge and month-to-month customers (realistic targeting bias).
- **T-learner CATE estimation**: Separate random forest models for control and treatment arms; CATE = E[Y|X, T=0] - E[Y|X, T=1].
- **Four-segment customer stratification**:
  - **Persuadables** (92 customers): Moderate baseline churn (~22%), very high uplift (~92%). Sweet spot for targeting.
  - **Sleeping Dogs** (441 customers): Moderate baseline (~20%), high uplift (~89%). Secondary targets.
  - **Sure Things** (262 customers): Low baseline churn (~2%), low uplift. Already retained; wasteful to spend on.
  - **Lost Causes** (202 customers): Very high baseline (~80%), minimal uplift (~3%). Unrecoverable; ROI negative.

#### Phase 5: Decision Dashboard
- **App**: `dashboard/app.py` (Dash + Plotly)
- **Headline KPI**: ROI comparison shows targeted spend (Persuadables + Sleeping Dogs) achieves **47.5% better ROI** than blanket discounting.
- **Customer Table**: Sortable, filterable by segment and contract; displays churn probability, uplift, and expected outcome if treated.
- **Customer Detail View**: SHAP-based explanation of why each customer is at risk.

#### Phase 6: Production Considerations
- **Model versioning**: Saved models via `joblib` to `models/`; retrain script in `retrain.py`.
- **MLflow**: Mentioned in requirements for future enhancement (logging hyperparameters, metrics, and model artifacts).
- **Retraining**: `python retrain.py` retrains the baseline model end-to-end.

---

## File Structure

```
stage/
├── data/
│   ├── telco_clean.csv               # Cleaned data (TotalCharges coerced, 11 rows dropped)
│   ├── telco_features_phase1.csv    # Features for modeling
│   └── eda/
│       ├── phase1_churn_breakdowns.png
│       └── phase2_calibration_curve.png
├── models/
│   └── baseline_churn_model.joblib    # Trained XGBoost + SMOTE pipeline
├── notebooks/
│   └── eda.ipynb                      # Exploratory data analysis
├── src/
│   ├── __init__.py
│   ├── data_prep.py                   # Data loading & cleaning
│   ├── features.py                    # Feature engineering
│   ├── model.py                       # Baseline churn classifier
│   ├── explain.py                     # SHAP explainability
│   ├── uplift.py                      # T-learner & customer segmentation
│   └── dashboard_data.py              # Dashboard data pipeline
├── dashboard/
│   └── app.py                         # Dash decision-support interface
├── requirements.txt                   # Dependencies
├── retrain.py                         # Model retraining script
└── README.md                          # This file
```

---

## Key Insights

### Imbalance Handling
- Target class (Churn) is ~26.5% of the dataset.
- SMOTE applied **only inside training folds** to avoid data leakage.
- Evaluation metrics prioritize ROC-AUC and precision/recall over accuracy.

### Causal Modeling
- **Realistic treatment bias**: Discount targeted to high-risk segments (high monthly charge, month-to-month contract) simulates real business behavior.
- **Heterogeneous treatment effects**: CATE learned separately per segment reveals who benefits most from intervention.
- **ROI focus**: Targeted spending (Persuadables + Sleeping Dogs) saves 47.5% more per dollar than blanket spend.

### Dashboard as Decision Support
- The ROI comparison is the headline: show it first and prominently.
- Filters allow exploration by segment and contract type.
- SHAP explanations provide accountability: "Why are we targeting this customer?"

---

## Running the System

### Train a fresh model
```bash
python retrain.py
```

### Explore data
```bash
jupyter notebook notebooks/eda.ipynb
```

### Start the dashboard
```bash
python dashboard/app.py
```

Navigate to **http://127.0.0.1:8050**.

---

## Dependencies

- `pandas`, `numpy`: Data manipulation.
- `scikit-learn`: Preprocessing, model evaluation, cross-validation.
- `xgboost`: Baseline classifier.
- `imbalanced-learn`: SMOTE for class imbalance.
- `shap`: Feature importance and customer explanations.
- `dash`, `plotly`: Interactive decision dashboard.
- `joblib`: Model serialization.
- `jupyter`: Exploratory analysis.

---

## Next Steps / Extensions

1. **MLflow integration**: Log hyperparameters, metrics, and model versions for better production tracking.
2. **Real experiment data**: Replace synthetic treatment assignment with actual A/B test results for stronger causal claims.
3. **Feature selection**: Use SHAP values to prune less-important features and simplify the model.
4. **Segment-specific models**: Train separate churn classifiers for each segment (Persuadables, etc.) for better segment-level predictions.
5. **Automated retraining**: Orchestrate retraining on a schedule or when model performance degrades.

---

## Authors & License

Built as a demonstration of causal inference for retention targeting. Use at your own discretion.

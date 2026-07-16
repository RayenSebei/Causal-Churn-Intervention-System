# Causal Churn Intervention System

Decision-support tool for **retention budget allocation**, not just a churn classifier.
Given a Telco-style customer base, the system estimates who is likely to churn, who is
likely to respond to a retention offer (uplift / CATE), and where budget should go.

## Quick start

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train baseline model
python retrain.py

# Run causal uplift pipeline
python -m src.uplift

# Launch dashboard
python dashboard/app.py
```

Open [http://127.0.0.1:8050](http://127.0.0.1:8050).

## Architecture

```
src/
  config.py            # Paths, model, causal, campaign settings
  constants.py         # Column names, segment labels, colors
  logging_config.py
  preprocessing.py     # Clean Telco data
  features.py          # Tenure buckets, service counts, ratios
  training.py          # XGBoost + SMOTE baseline training
  evaluation.py        # Metrics and diagnostic plots
  prediction.py        # Load model and score customers
  explainability.py    # SHAP public API (wraps explain.py)
  dashboard_data.py    # Join predictions, uplift, explanations
  causal/
    learners.py        # T- / S- / X-learners
    treatment.py       # Synthetic treatment assignment
    simulation.py      # Outcome injection + uplift pipeline
    policy.py          # Segmentation and budget selection
    roi.py             # Campaign economics helpers
dashboard/
  app.py               # Thin Dash entry point
  layout.py
  callbacks.py
  metrics.py
  plots.py
  filters.py
tests/
```

Legacy shims (`src.model`, `src.uplift`, `src.data_prep`) re-export the new modules
so older imports keep working.

## Segments (targeting policy)

| Segment | Meaning | Target? |
|---|---|---|
| Persuadables | High churn risk, positive uplift | Yes |
| Low-Risk Upside | Lower baseline risk, positive uplift | Yes |
| Sleeping Dogs | Negative CATE — treatment can backfire | **No** |
| Sure Things | Low risk, low uplift | No |
| Lost Causes | High risk, low uplift | No |

## Tests

```bash
python -m pytest tests/ -q
```

## Data

Uses the public Telco Customer Churn CSV at the project root
(`WA_Fn-UseC_-Telco-Customer-Churn.csv`). Cleaned / featured artefacts are written
under `data/`; the trained model under `models/`.

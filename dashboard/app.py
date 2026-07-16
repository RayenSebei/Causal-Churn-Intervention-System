"""Dash entry point for the retention decision-support dashboard."""

from __future__ import annotations

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import dash

from dashboard.callbacks import register_callbacks
from dashboard.layout import build_layout
from dashboard.metrics import compute_roi_metrics
from src.config import paths
from src.dashboard_data import load_dashboard_data

DATA_PATH = paths.raw_csv
MODEL_PATH = paths.baseline_model

dashboard_data = load_dashboard_data(DATA_PATH, MODEL_PATH)
df = dashboard_data["df"].reset_index(drop=True)
roi_metrics = compute_roi_metrics(df)
uplift_meta = dashboard_data["uplift_results"]

app = dash.Dash(__name__)
app.layout = build_layout(df, roi_metrics, uplift_meta)
register_callbacks(app, df)


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)

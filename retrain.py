#!/usr/bin/env python3
"""Retrain the baseline churn model end-to-end."""

from pathlib import Path

from src.model import train_baseline_model


def main():
    base_dir = Path(__file__).parent
    csv_path = base_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    model_output_path = base_dir / "models" / "baseline_churn_model.joblib"
    calibration_output_path = base_dir / "data" / "eda" / "phase2_calibration_curve.png"

    print("[*] Training baseline churn model...")
    metrics, artifacts = train_baseline_model(
        csv_path,
        model_output_path=model_output_path,
        calibration_output_path=calibration_output_path,
    )

    print(f"\n[OK] Model trained and saved to {model_output_path}")
    print(f"[OK] Calibration curve saved to {calibration_output_path}")
    print(f"\n[Metrics]\n{metrics}")
    print(f"\n[Classification Report]\n{artifacts['classification_report']}")


if __name__ == "__main__":
    main()

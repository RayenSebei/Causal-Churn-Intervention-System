#!/usr/bin/env python3
"""Retrain the baseline churn model end-to-end."""

from src.config import paths
from src.training import train_baseline_model


def main():
    csv_path = paths.raw_csv
    model_output_path = paths.baseline_model
    calibration_output_path = paths.eda_dir / "phase2_calibration_curve.png"

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

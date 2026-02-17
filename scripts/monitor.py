# ============================================================
# scripts/monitor.py
# Drift monitoring script for the Churn Prediction SaaS.
# Compares production data against training data to detect
# feature drift, prediction drift, and performance decay.
# ============================================================

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from loguru import logger
from datetime import datetime

from src.utils.helpers import load_config, setup_logging
from src.monitoring.drift import DriftDetector
from src.features.engineer import FeatureEngineer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run drift monitoring against production data"
    )
    parser.add_argument(
        "--data", "-d", type=str, required=True,
        help="Path to the production/new data CSV file",
    )
    parser.add_argument(
        "--reference", "-r", type=str, default=None,
        help="Path to reference (training) data CSV. Defaults to configured raw data path.",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="reports/monitoring",
        help="Output directory for drift reports (default: reports/monitoring)",
    )
    parser.add_argument(
        "--baseline-metric", type=float, default=None,
        help="Baseline ROC-AUC to check for performance decay",
    )
    return parser.parse_args()


def run_monitoring(args):
    """Execute the drift monitoring pipeline."""
    # Load config and set up logging
    config = load_config()
    setup_logging(config.get("logging", {}).get("dir", "logs"))

    logger.info("=" * 60)
    logger.info("DRIFT MONITORING PIPELINE")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Production data: {args.data}")
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # Step 1: Load Data
    # ----------------------------------------------------------
    logger.info("Step 1: Loading data...")

    # Load production data
    prod_path = Path(args.data)
    if not prod_path.exists():
        logger.error(f"Production data not found: {prod_path}")
        return
    prod_df = pd.read_csv(prod_path)
    logger.info(f"Production data: {prod_df.shape[0]} rows, {prod_df.shape[1]} columns")

    # Load reference data
    if args.reference:
        ref_path = Path(args.reference)
    else:
        ref_path = Path(config.get("data", {}).get("raw_path", "data/raw/telco_churn.csv"))

    if not ref_path.exists():
        logger.error(f"Reference data not found: {ref_path}")
        return
    ref_df = pd.read_csv(ref_path)
    logger.info(f"Reference data: {ref_df.shape[0]} rows, {ref_df.shape[1]} columns")

    # ----------------------------------------------------------
    # Step 2: Initialize Detector
    # ----------------------------------------------------------
    logger.info("Step 2: Initializing drift detector...")
    detector = DriftDetector(config, output_dir=args.output)

    # Set reference distributions from training data (numeric only)
    ref_numeric = ref_df.select_dtypes(include=[np.number])
    detector.set_reference(ref_numeric)
    logger.info(f"Reference distributions set for {len(detector.reference_distributions)} features")

    # ----------------------------------------------------------
    # Step 3: Feature Drift Check
    # ----------------------------------------------------------
    logger.info("Step 3: Checking feature drift...")
    prod_numeric = prod_df.select_dtypes(include=[np.number])
    drift_report = detector.check_feature_drift(prod_numeric)

    # Log results
    drifted = [f for f, r in drift_report.items() if r["status"] == "DRIFT_DETECTED"]
    warned = [f for f, r in drift_report.items() if r["status"] == "WARNING"]

    logger.info(f"  Drifted features: {len(drifted)}")
    if drifted:
        for f in drifted:
            logger.warning(f"    DRIFT: {f} (PSI={drift_report[f]['psi']:.4f})")
    logger.info(f"  Warning features: {len(warned)}")
    if warned:
        for f in warned:
            logger.info(f"    WARNING: {f} (PSI={drift_report[f]['psi']:.4f})")

    # ----------------------------------------------------------
    # Step 4: Prediction Drift Check
    # ----------------------------------------------------------
    logger.info("Step 4: Checking prediction drift...")
    model_path = Path("models/best_model.joblib")
    preprocessor_path = Path("models/preprocessor.joblib")
    selected_features_path = Path("models/selected_features.joblib")

    if model_path.exists() and preprocessor_path.exists():
        try:
            model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)
            engineer = FeatureEngineer(config)

            # Get predictions on reference data
            ref_eng = engineer.engineer_all_features(ref_df.copy())
            target_col = config.get("features", {}).get("target", "Churn")
            if target_col in ref_eng.columns:
                ref_eng = ref_eng.drop(columns=[target_col], errors="ignore")
            id_col = config.get("data", {}).get("id_column", "customerID")
            if id_col in ref_eng.columns:
                ref_eng = ref_eng.drop(columns=[id_col], errors="ignore")
            ref_features = preprocessor.transform(ref_eng)

            selected = None
            selected_names = None
            selected_available = None
            if selected_features_path.exists():
                selected = joblib.load(selected_features_path)
                try:
                    try:
                        selected_names = preprocessor._get_feature_names()
                    except Exception:
                        selected_names = [f"feature_{i}" for i in range(ref_features.shape[1])]

                    if not any(f in selected_names for f in selected):
                        selected_names = [f"feature_{i}" for i in range(ref_features.shape[1])]

                    ref_df_feat = pd.DataFrame(ref_features, columns=selected_names)
                    selected_available = [f for f in selected if f in ref_df_feat.columns]
                    if selected_available:
                        ref_features = ref_df_feat[selected_available].values
                except Exception:
                    pass

            ref_preds = model.predict_proba(ref_features)[:, 1]

            # Get predictions on production data
            prod_eng = engineer.engineer_all_features(prod_df.copy())
            if target_col in prod_eng.columns:
                prod_eng = prod_eng.drop(columns=[target_col], errors="ignore")
            if id_col in prod_eng.columns:
                prod_eng = prod_eng.drop(columns=[id_col], errors="ignore")
            prod_features = preprocessor.transform(prod_eng)

            if selected is not None and selected_names is not None and selected_available is not None:
                try:
                    prod_df_feat = pd.DataFrame(prod_features, columns=selected_names)
                    if selected_available:
                        prod_features = prod_df_feat[selected_available].values
                except Exception:
                    pass

            prod_preds = model.predict_proba(prod_features)[:, 1]

            pred_drift = detector.check_prediction_drift(ref_preds, prod_preds)
            logger.info(f"  Prediction drift PSI: {pred_drift['psi']:.4f} [{pred_drift['status']}]")
            logger.info(f"  Reference mean: {pred_drift['ref_mean_prediction']:.4f}")
            logger.info(f"  Production mean: {pred_drift['cur_mean_prediction']:.4f}")

        except Exception as e:
            logger.warning(f"Prediction drift check failed: {e}")
    else:
        logger.info("  Skipped (model or preprocessor not found)")

    # ----------------------------------------------------------
    # Step 5: Performance Decay Check (if baseline provided)
    # ----------------------------------------------------------
    if args.baseline_metric is not None:
        logger.info("Step 5: Checking performance decay...")
        # This requires labeled production data
        target_col = config.get("features", {}).get("target", "Churn")
        if target_col in prod_df.columns:
            try:
                from sklearn.metrics import roc_auc_score
                y_true = prod_df[target_col]
                if y_true.dtype == object:
                    y_true = (y_true == "Yes").astype(int)
                current_auc = roc_auc_score(y_true, prod_preds)
                decay = detector.check_performance_decay(
                    args.baseline_metric, current_auc, "roc_auc"
                )
                logger.info(f"  Baseline: {decay['baseline_value']:.4f}")
                logger.info(f"  Current:  {decay['current_value']:.4f}")
                logger.info(f"  Status:   {decay['status']}")
            except Exception as e:
                logger.warning(f"Performance decay check failed: {e}")
        else:
            logger.info("  Skipped (no target column in production data)")
    else:
        logger.info("Step 5: Performance decay check skipped (no --baseline-metric provided)")

    # ----------------------------------------------------------
    # Step 6: Generate Reports
    # ----------------------------------------------------------
    logger.info("Step 6: Generating drift report plot...")
    try:
        plot_path = detector.plot_drift_report(drift_report)
        logger.info(f"  Drift report saved to: {plot_path}")
    except Exception as e:
        logger.warning(f"  Report generation failed: {e}")

    # Summary
    summary = detector.generate_monitoring_summary()
    logger.info("=" * 60)
    logger.info("MONITORING SUMMARY")
    logger.info(f"  Recommendation: {summary['recommendation']}")
    logger.info(f"  Action: {summary['action']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run_monitoring(args)

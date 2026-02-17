# ============================================================
# scripts/retrain.py
# Retraining scheduler for the Churn Prediction SaaS.
# Re-runs the training pipeline on updated data, compares
# performance with the current production model, and
# optionally promotes the new model if it outperforms.
# ============================================================

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import json
import shutil
from loguru import logger
from datetime import datetime
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils.helpers import load_config, setup_logging, load_model, save_model
from src.data.validator import DataValidator
from src.data.preprocessor import DataPreprocessor
from src.features.engineer import FeatureEngineer
from src.features.selector import FeatureSelector
from src.models.trainer import ModelTrainer
from src.models.evaluator import ModelEvaluator


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Retrain the churn prediction model with new data"
    )
    parser.add_argument(
        "--data", "-d", type=str, required=True,
        help="Path to the updated training data CSV",
    )
    parser.add_argument(
        "--auto-promote", action="store_true",
        help="Automatically promote the new model if it outperforms the current one",
    )
    parser.add_argument(
        "--min-improvement", type=float, default=0.005,
        help="Minimum ROC-AUC improvement required for promotion (default: 0.005)",
    )
    parser.add_argument(
        "--backup", action="store_true", default=True,
        help="Backup current model before replacing (default: True)",
    )
    return parser.parse_args()


def backup_current_model():
    """Backup the current production model artifacts."""
    backup_dir = Path("models/backup") / datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir.mkdir(parents=True, exist_ok=True)

    artifacts = [
        "models/best_model.joblib",
        "models/preprocessor.joblib",
        "models/selected_features.joblib",
        "models/optimal_threshold.joblib",
        "models/registry_index.json",
    ]

    for artifact in artifacts:
        src = Path(artifact)
        if src.exists():
            shutil.copy2(src, backup_dir / src.name)
            logger.info(f"  Backed up: {src.name}")

    logger.info(f"Model backup saved to: {backup_dir}")
    return str(backup_dir)


def get_current_performance():
    """Get the current production model's ROC-AUC from the registry."""
    registry_path = Path("models/registry_index.json")
    if registry_path.exists():
        with open(registry_path) as f:
            registry = json.load(f)
        for key, info in registry.get("models", {}).items():
            metrics = info.get("metrics", {})
            if "roc_auc" in metrics:
                return float(metrics["roc_auc"])
    return 0.0


def run_retraining(args):
    """Execute the retraining pipeline."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("dir", "logs"))

    logger.info("=" * 60)
    logger.info("RETRAINING PIPELINE")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Data source: {args.data}")
    logger.info(f"Auto-promote: {args.auto_promote}")
    logger.info(f"Min improvement: {args.min_improvement}")
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # Step 1: Load and Validate New Data
    # ----------------------------------------------------------
    logger.info("Step 1: Loading and validating new data...")
    data_path = Path(args.data)
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return

    df = pd.read_csv(data_path)
    logger.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")

    validator = DataValidator(config)
    validation = validator.run_all_validations(df)
    failed = [k for k, v in validation.items() if not v]
    if failed:
        logger.warning(f"Validation warnings: {failed}")

    # ----------------------------------------------------------
    # Step 2: Feature Engineering
    # ----------------------------------------------------------
    logger.info("Step 2: Engineering features...")
    engineer = FeatureEngineer(config)
    df_featured = engineer.engineer_all_features(df)
    logger.info(f"Features after engineering: {df_featured.shape[1]} columns")

    # ----------------------------------------------------------
    # Step 3: Preprocessing
    # ----------------------------------------------------------
    logger.info("Step 3: Preprocessing...")
    preprocessor = DataPreprocessor(config)

    target_col = config.get("features", {}).get("target", "Churn")
    if df_featured[target_col].dtype == object:
        df_featured[target_col] = (df_featured[target_col] == "Yes").astype(int)

    X_train_df, X_test_df, y_train, y_test = preprocessor.split_data(df_featured)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # ----------------------------------------------------------
    # Step 4: Feature Selection
    # ----------------------------------------------------------
    logger.info("Step 4: Selecting features...")
    selector = FeatureSelector(config)
    try:
        feature_names = preprocessor._get_feature_names()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    X_train_named = pd.DataFrame(X_train, columns=feature_names)
    X_test_named = pd.DataFrame(X_test, columns=feature_names)

    selected_features = selector.select_features(X_train_named, y_train)
    logger.info(f"Selected {len(selected_features)} features")

    X_train_filtered, _ = selector.select_features_df(X_train_named, y_train)
    X_train_selected = X_train_filtered.values
    X_test_selected = X_test_named[selected_features].values

    # ----------------------------------------------------------
    # Step 5: Model Training
    # ----------------------------------------------------------
    logger.info("Step 5: Training models...")
    trainer = ModelTrainer(config)
    trained_models = trainer.train_all_models(X_train_selected, y_train)
    logger.info(f"Trained {len(trained_models)} models")

    # ----------------------------------------------------------
    # Step 6: Evaluation
    # ----------------------------------------------------------
    logger.info("Step 6: Evaluating models...")
    evaluator = ModelEvaluator(config)

    all_results = {}
    for name, model in trained_models.items():
        metrics = evaluator.compute_metrics(model, X_test_selected, y_test)
        all_results[name] = metrics
        logger.info(f"  {name}: ROC-AUC={metrics.get('roc_auc', 0):.4f}")

    # Find best new model
    best_name = max(all_results, key=lambda k: all_results[k].get("roc_auc", 0))
    best_model = trained_models[best_name]
    new_roc_auc = all_results[best_name].get("roc_auc", 0)
    logger.info(f"Best new model: {best_name} (ROC-AUC: {new_roc_auc:.4f})")

    # ----------------------------------------------------------
    # Step 7: Compare with Current Model
    # ----------------------------------------------------------
    logger.info("Step 7: Comparing with current model...")
    current_roc_auc = get_current_performance()
    improvement = new_roc_auc - current_roc_auc

    logger.info(f"  Current production ROC-AUC: {current_roc_auc:.4f}")
    logger.info(f"  New model ROC-AUC:          {new_roc_auc:.4f}")
    logger.info(f"  Improvement:                {improvement:+.4f}")

    # ----------------------------------------------------------
    # Step 8: Promote or Hold
    # ----------------------------------------------------------
    should_promote = improvement >= args.min_improvement

    if should_promote:
        logger.info("New model OUTPERFORMS current model.")

        if args.auto_promote:
            logger.info("Step 8: Auto-promoting new model...")

            if args.backup:
                backup_path = backup_current_model()

            # Find optimal threshold
            try:
                proba = best_model.predict_proba(X_test_selected)[:, 1]
                opt_thr, _ = trainer.find_optimal_threshold(y_test, proba)
            except Exception:
                opt_thr = 0.5

            # Save new artifacts
            save_model(best_model, "models/best_model.joblib")
            joblib.dump(preprocessor, "models/preprocessor.joblib")
            joblib.dump(selected_features, "models/selected_features.joblib")
            joblib.dump(opt_thr, "models/optimal_threshold.joblib")

            logger.info("New model promoted to production.")
            logger.info(f"  Model: {best_name}")
            logger.info(f"  ROC-AUC: {new_roc_auc:.4f}")
            logger.info(f"  Threshold: {opt_thr:.3f}")
        else:
            logger.info("Auto-promote is OFF. Run with --auto-promote to deploy.")
            # Save as challenger for A/B testing
            challenger_path = f"models/challenger_{best_name}_{datetime.now().strftime('%Y%m%d')}.joblib"
            save_model(best_model, challenger_path)
            logger.info(f"Challenger model saved to: {challenger_path}")
            logger.info(
                "Run A/B test with:\n"
                f"  python scripts/ab_test.py --challenger {challenger_path} --data {args.data}"
            )
    else:
        logger.info(
            f"New model does NOT meet minimum improvement threshold ({args.min_improvement})."
        )
        logger.info("Current production model retained.")

    # Summary
    logger.info("=" * 60)
    logger.info("RETRAINING SUMMARY")
    logger.info(f"  Data: {args.data} ({df.shape[0]} rows)")
    logger.info(f"  Best model: {best_name}")
    logger.info(f"  New ROC-AUC: {new_roc_auc:.4f}")
    logger.info(f"  Current ROC-AUC: {current_roc_auc:.4f}")
    logger.info(f"  Improvement: {improvement:+.4f}")
    logger.info(f"  Promoted: {'Yes' if (should_promote and args.auto_promote) else 'No'}")
    logger.info("=" * 60)


if __name__ == "__main__":
    args = parse_args()
    run_retraining(args)

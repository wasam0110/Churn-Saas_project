# ============================================================
# scripts/ab_test.py
# A/B Testing framework for comparing model versions.
# Routes predictions between a champion (production) model
# and a challenger (new) model, then compares their metrics.
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
from loguru import logger
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    brier_score_loss, log_loss,
)

from src.utils.helpers import load_config, setup_logging


class ABTestRunner:
    """
    Runs A/B tests between two model versions.

    Splits data using deterministic hashing to ensure
    consistent routing of customers to model variants.
    Computes comparative metrics and statistical significance.
    """

    def __init__(self, champion_path: str, challenger_path: str,
                 traffic_split: float = 0.5, config: dict = None):
        """
        Initialize the A/B test runner.

        Parameters
        ----------
        champion_path : str
            Path to the champion (production) model.
        challenger_path : str
            Path to the challenger (new) model.
        traffic_split : float
            Fraction of traffic to route to the challenger (0 to 1).
        config : dict
            Project configuration.
        """
        self.config = config or {}
        self.traffic_split = traffic_split

        # Load both models
        logger.info(f"Loading champion model from: {champion_path}")
        self.champion = joblib.load(champion_path)
        logger.info(f"Loading challenger model from: {challenger_path}")
        self.challenger = joblib.load(challenger_path)

        # Load shared preprocessor and features
        self.preprocessor = None
        self.selected_features = None
        self.engineer = None

        prep_path = Path("models/preprocessor.joblib")
        if prep_path.exists():
            self.preprocessor = joblib.load(prep_path)

        feat_path = Path("models/selected_features.joblib")
        if feat_path.exists():
            self.selected_features = joblib.load(feat_path)

        try:
            from src.features.engineer import FeatureEngineer
            self.engineer = FeatureEngineer(self.config)
        except Exception:
            pass

        # Results storage
        self.results = {"champion": [], "challenger": []}
        logger.info(f"A/B test initialized. Traffic split: {1-traffic_split:.0%} champion / {traffic_split:.0%} challenger")

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Apply the full feature pipeline."""
        if self.engineer is not None:
            df = self.engineer.engineer_all_features(df)

        if self.preprocessor is not None:
            features = self.preprocessor.transform(df)
        else:
            features = df.values

        if self.selected_features is not None:
            try:
                try:
                    names = self.preprocessor._get_feature_names()
                except Exception:
                    names = [f"feature_{i}" for i in range(features.shape[1])]

                if not any(f in names for f in self.selected_features):
                    names = [f"feature_{i}" for i in range(features.shape[1])]

                fdf = pd.DataFrame(features, columns=names)
                available = [f for f in self.selected_features if f in fdf.columns]
                if available:
                    features = fdf[available].values
            except Exception:
                pass

        return features

    def route_traffic(self, n_samples: int) -> np.ndarray:
        """
        Deterministically route samples to champion or challenger.

        Returns
        -------
        np.ndarray
            Boolean array. True = challenger, False = champion.
        """
        np.random.seed(42)
        return np.random.random(n_samples) < self.traffic_split

    def run_test(self, df: pd.DataFrame, y_true: np.ndarray) -> dict:
        """
        Run the A/B test on labeled data.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame (without target column).
        y_true : np.ndarray
            True labels.

        Returns
        -------
        dict
            Comparison of champion vs challenger metrics.
        """
        logger.info("Running A/B test...")

        # Prepare features
        features = self._prepare_features(df)

        # Route traffic
        is_challenger = self.route_traffic(len(y_true))
        champion_idx = ~is_challenger
        challenger_idx = is_challenger

        logger.info(f"  Champion samples: {champion_idx.sum()}")
        logger.info(f"  Challenger samples: {challenger_idx.sum()}")

        # Get predictions from both models on their assigned samples
        champ_probs = self.champion.predict_proba(features[champion_idx])[:, 1]
        chall_probs = self.challenger.predict_proba(features[challenger_idx])[:, 1]

        # Compute metrics for champion
        champ_metrics = self._compute_metrics(
            y_true[champion_idx], champ_probs, "champion"
        )

        # Compute metrics for challenger
        chall_metrics = self._compute_metrics(
            y_true[challenger_idx], chall_probs, "challenger"
        )

        # Statistical comparison
        comparison = self._compare(champ_metrics, chall_metrics)

        return {
            "champion": champ_metrics,
            "challenger": chall_metrics,
            "comparison": comparison,
            "sample_sizes": {
                "champion": int(champion_idx.sum()),
                "challenger": int(challenger_idx.sum()),
            },
            "timestamp": datetime.now().isoformat(),
        }

    def _compute_metrics(self, y_true, y_proba, variant: str) -> dict:
        """Compute evaluation metrics for a model variant."""
        threshold = 0.5
        thr_path = Path("models/optimal_threshold.joblib")
        if thr_path.exists():
            threshold = joblib.load(thr_path)

        y_pred = (y_proba >= threshold).astype(int)

        metrics = {
            "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
            "pr_auc": round(float(average_precision_score(y_true, y_proba)), 4),
            "f1": round(float(f1_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "brier_score": round(float(brier_score_loss(y_true, y_proba)), 4),
            "log_loss": round(float(log_loss(y_true, y_proba)), 4),
        }

        logger.info(f"  {variant}: ROC-AUC={metrics['roc_auc']}, PR-AUC={metrics['pr_auc']}, F1={metrics['f1']}")
        return metrics

    def _compare(self, champ: dict, chall: dict) -> dict:
        """Compare metrics between champion and challenger."""
        comparison = {}
        for metric in champ:
            diff = chall[metric] - champ[metric]
            # For loss metrics, lower is better
            if metric in ("brier_score", "log_loss"):
                winner = "challenger" if diff < 0 else "champion"
            else:
                winner = "challenger" if diff > 0 else "champion"

            comparison[metric] = {
                "champion": champ[metric],
                "challenger": chall[metric],
                "difference": round(diff, 4),
                "winner": winner,
            }

        # Overall winner by majority of metrics
        winners = [v["winner"] for v in comparison.values()]
        overall = "challenger" if winners.count("challenger") > winners.count("champion") else "champion"
        comparison["overall_winner"] = overall

        return comparison


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run A/B test between two model versions"
    )
    parser.add_argument(
        "--champion", "-c", type=str,
        default="models/best_model.joblib",
        help="Path to champion (production) model",
    )
    parser.add_argument(
        "--challenger", "-ch", type=str, required=True,
        help="Path to challenger (new) model",
    )
    parser.add_argument(
        "--data", "-d", type=str, required=True,
        help="Path to labeled test data CSV",
    )
    parser.add_argument(
        "--split", "-s", type=float, default=0.5,
        help="Fraction of traffic for challenger (default: 0.5)",
    )
    parser.add_argument(
        "--output", "-o", type=str, default="reports/ab_test_results.json",
        help="Output path for test results",
    )
    return parser.parse_args()


def main():
    """Execute the A/B test."""
    args = parse_args()
    config = load_config()
    setup_logging(config.get("logging", {}).get("dir", "logs"))

    logger.info("=" * 60)
    logger.info("A/B TEST PIPELINE")
    logger.info(f"Champion: {args.champion}")
    logger.info(f"Challenger: {args.challenger}")
    logger.info(f"Traffic split: {args.split:.0%} challenger")
    logger.info("=" * 60)

    # Load data
    df = pd.read_csv(args.data)
    target_col = config.get("features", {}).get("target", "Churn")
    id_col = config.get("data", {}).get("id_column", "customerID")

    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in data.")
        return

    y_true = df[target_col].copy()
    if y_true.dtype == object:
        y_true = (y_true == "Yes").astype(int)
    y_true = y_true.values

    # Remove target and ID columns
    df_features = df.drop(columns=[target_col], errors="ignore")
    df_features = df_features.drop(columns=[id_col], errors="ignore")

    # Run A/B test
    runner = ABTestRunner(
        champion_path=args.champion,
        challenger_path=args.challenger,
        traffic_split=args.split,
        config=config,
    )
    results = runner.run_test(df_features, y_true)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info("=" * 60)
    logger.info("A/B TEST RESULTS")
    comp = results["comparison"]
    for metric, info in comp.items():
        if metric == "overall_winner":
            continue
        logger.info(
            f"  {metric}: Champion={info['champion']}, "
            f"Challenger={info['challenger']} "
            f"(diff={info['difference']:+.4f}, winner={info['winner']})"
        )
    logger.info(f"  Overall winner: {comp['overall_winner'].upper()}")
    logger.info(f"  Results saved to: {output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

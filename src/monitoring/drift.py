# ============================================================
# src/monitoring/drift.py
# Data and model drift detection using Population Stability
# Index (PSI) and prediction distribution monitoring.
# ============================================================

import numpy as np                                 # NumPy for numerical operations
import pandas as pd                                # Pandas for DataFrames
from scipy import stats                            # SciPy for statistical tests
import matplotlib.pyplot as plt                    # Matplotlib for plotting
plt.switch_backend('Agg')                          # Use non-interactive backend for server environments
from loguru import logger                          # Structured logging
from typing import Dict, Any, List, Optional, Tuple  # Type hints
from pathlib import Path                           # Object-oriented paths
from datetime import datetime                      # Timestamps for monitoring logs


class DriftDetector:
    """
    Detects data drift and model performance degradation.

    Uses Population Stability Index (PSI) to measure how much
    the distribution of features has shifted from training to production.

    PSI interpretation:
    - PSI < 0.1:  No significant drift (stable)
    - 0.1 ≤ PSI < 0.2:  Moderate drift (investigate)
    - PSI ≥ 0.2:  Significant drift (action needed, consider retraining)
    """

    def __init__(self, config: dict, output_dir: str = "reports/monitoring"):
        """
        Initialize the DriftDetector with configuration.

        Parameters
        ----------
        config : dict
            Project configuration dictionary.
        output_dir : str
            Directory for saving monitoring reports and plots.
        """
        # Store the configuration
        self.config = config
        # Extract the PSI alert threshold from config
        self.psi_threshold = config["monitoring"]["psi_threshold"]
        # Extract the performance decay threshold
        self.performance_threshold = config["monitoring"]["performance_decay_threshold"]
        # Set up the output directory
        self.output_dir = Path(output_dir)
        # Create the directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Store reference (training) distributions for comparison
        self.reference_distributions: Dict[str, np.ndarray] = {}
        # Store monitoring history for trend analysis
        self.monitoring_history: List[Dict] = []
        # Log initialization
        logger.info(f"DriftDetector initialized. PSI threshold: {self.psi_threshold}")

    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute the Population Stability Index (PSI) between
        a reference distribution and a current distribution.

        PSI = Σ (P_i - Q_i) × ln(P_i / Q_i)
        where P = reference proportions, Q = current proportions.

        Parameters
        ----------
        reference : np.ndarray
            Reference (training) distribution values.
        current : np.ndarray
            Current (production) distribution values.
        n_bins : int
            Number of bins for discretizing continuous distributions.

        Returns
        -------
        float
            The PSI value (0 = identical, higher = more drift).
        """
        # Remove NaN values from both arrays
        reference = reference[~np.isnan(reference)]
        current = current[~np.isnan(current)]

        # Create bin edges from the reference distribution
        # This ensures consistent binning between reference and current
        bin_edges = np.percentile(
            reference,                                 # Source for bin edges
            np.linspace(0, 100, n_bins + 1),          # Equal-frequency percentiles
        )
        # Ensure unique bin edges (avoid zero-width bins)
        bin_edges = np.unique(bin_edges)
        # Need at least 2 edges to form 1 bin
        if len(bin_edges) < 2:
            return 0.0

        # Compute bin counts for reference distribution
        ref_counts, _ = np.histogram(reference, bins=bin_edges)
        # Compute bin counts for current distribution
        cur_counts, _ = np.histogram(current, bins=bin_edges)

        # Convert counts to proportions, adding a small epsilon to avoid log(0)
        epsilon = 1e-6                                 # Small constant for numerical stability
        ref_proportions = (ref_counts / len(reference)) + epsilon
        cur_proportions = (cur_counts / len(current)) + epsilon

        # Compute PSI using the formula: Σ (P_i - Q_i) × ln(P_i / Q_i)
        psi = np.sum(
            (cur_proportions - ref_proportions) *      # Difference in proportions
            np.log(cur_proportions / ref_proportions)  # Log ratio of proportions
        )

        # Return the PSI value
        return round(float(psi), 6)

    def set_reference(self, X_train: pd.DataFrame) -> None:
        """
        Set the reference distributions from training data.

        This should be called once after training, storing the
        training distributions for future comparison.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature DataFrame.
        """
        # Store the distribution of each numeric column
        for col in X_train.select_dtypes(include=[np.number]).columns:
            # Store the column values as a numpy array
            self.reference_distributions[col] = X_train[col].values.copy()
            # Log each stored distribution
            logger.debug(f"Reference distribution stored for: {col}")

        # Log the total number of features tracked
        logger.info(f"Reference distributions set for {len(self.reference_distributions)} features")

    def check_feature_drift(
        self,
        X_current: pd.DataFrame,
    ) -> Dict[str, Dict]:
        """
        Check all features for drift against the reference distributions.

        Parameters
        ----------
        X_current : pd.DataFrame
            Current (production) feature DataFrame.

        Returns
        -------
        dict
            Dictionary mapping feature names to drift analysis results.
        """
        # Check that reference distributions exist
        if not self.reference_distributions:
            raise RuntimeError("No reference distributions set. Call set_reference() first.")

        # Initialize the drift report
        drift_report = {}
        # Count features with significant drift
        n_drifted = 0

        # Log the start of drift checking
        logger.info("Checking feature drift...")

        # Check each feature's distribution
        for col, ref_values in self.reference_distributions.items():
            # Check if this feature exists in the current data
            if col not in X_current.columns:
                logger.warning(f"Feature '{col}' missing from current data")
                continue

            # Get current values
            cur_values = X_current[col].values

            # Compute PSI for this feature
            psi = self.compute_psi(ref_values, cur_values)

            # Determine drift status based on PSI thresholds
            if psi >= self.psi_threshold:
                status = "DRIFT_DETECTED"              # Significant drift — action needed
                n_drifted += 1
            elif psi >= 0.1:
                status = "WARNING"                     # Moderate drift — investigate
            else:
                status = "STABLE"                      # No significant drift

            # Also compute Kolmogorov-Smirnov test for additional validation
            ks_statistic, ks_pvalue = stats.ks_2samp(ref_values, cur_values)

            # Store the results for this feature
            drift_report[col] = {
                "psi": psi,                            # Population Stability Index
                "status": status,                      # Drift categorization
                "ks_statistic": round(float(ks_statistic), 4),  # KS test statistic
                "ks_pvalue": round(float(ks_pvalue), 6),        # KS test p-value
                "ref_mean": round(float(np.nanmean(ref_values)), 4),  # Reference mean
                "cur_mean": round(float(np.nanmean(cur_values)), 4),  # Current mean
                "ref_std": round(float(np.nanstd(ref_values)), 4),    # Reference std
                "cur_std": round(float(np.nanstd(cur_values)), 4),    # Current std
            }

            # Log features with drift
            if status != "STABLE":
                logger.warning(f"  {col}: PSI={psi:.4f} [{status}]")

        # Log the summary
        logger.info(f"Drift check complete: {n_drifted}/{len(drift_report)} features drifted")

        # Add timestamp and store in monitoring history
        self.monitoring_history.append({
            "timestamp": datetime.now().isoformat(),
            "features_checked": len(drift_report),
            "features_drifted": n_drifted,
            "drift_report": drift_report,
        })

        # Return the drift report
        return drift_report

    def check_prediction_drift(
        self,
        reference_predictions: np.ndarray,
        current_predictions: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Check if the model's prediction distribution has shifted.

        A shift in predictions (even without feature drift) may indicate
        a concept drift — the relationship between features and target
        has changed.

        Parameters
        ----------
        reference_predictions : np.ndarray
            Predictions from the reference period (e.g., training/validation).
        current_predictions : np.ndarray
            Predictions from the current production period.

        Returns
        -------
        dict
            Dictionary with PSI, statistics, and drift status.
        """
        # Compute PSI for prediction distributions
        psi = self.compute_psi(reference_predictions, current_predictions)

        # Compute KS test
        ks_stat, ks_pval = stats.ks_2samp(reference_predictions, current_predictions)

        # Determine status
        if psi >= self.psi_threshold:
            status = "PREDICTION_DRIFT"
        elif psi >= 0.1:
            status = "WARNING"
        else:
            status = "STABLE"

        # Build the result
        result = {
            "psi": psi,                                # PSI for predictions
            "status": status,                          # Drift status
            "ks_statistic": round(float(ks_stat), 4),  # KS test statistic
            "ks_pvalue": round(float(ks_pval), 6),     # KS p-value
            "ref_mean_prediction": round(float(np.mean(reference_predictions)), 4),
            "cur_mean_prediction": round(float(np.mean(current_predictions)), 4),
            "ref_std_prediction": round(float(np.std(reference_predictions)), 4),
            "cur_std_prediction": round(float(np.std(current_predictions)), 4),
        }

        # Log the result
        logger.info(f"Prediction drift check: PSI={psi:.4f} [{status}]")

        # Return the result
        return result

    def check_performance_decay(
        self,
        baseline_metric: float,
        current_metric: float,
        metric_name: str = "pr_auc",
    ) -> Dict[str, Any]:
        """
        Check if model performance has decayed beyond the threshold.

        Parameters
        ----------
        baseline_metric : float
            The metric value at deployment time.
        current_metric : float
            The current metric value.
        metric_name : str
            Name of the metric being compared.

        Returns
        -------
        dict
            Dictionary with decay analysis.
        """
        # Calculate the absolute drop in performance
        drop = baseline_metric - current_metric
        # Calculate the relative (percentage) drop
        relative_drop = drop / baseline_metric if baseline_metric > 0 else 0

        # Determine if performance has decayed beyond threshold
        if relative_drop >= self.performance_threshold:
            status = "PERFORMANCE_DECAY"               # Significant decay
        elif relative_drop >= self.performance_threshold / 2:
            status = "WARNING"                         # Moderate decay
        else:
            status = "STABLE"                          # Performance is holding

        # Build the result
        result = {
            "metric_name": metric_name,
            "baseline_value": round(baseline_metric, 4),
            "current_value": round(current_metric, 4),
            "absolute_drop": round(drop, 4),
            "relative_drop": round(relative_drop, 4),
            "status": status,
            "retrain_recommended": status == "PERFORMANCE_DECAY",
        }

        # Log the result
        logger.info(
            f"Performance check ({metric_name}): "
            f"{baseline_metric:.4f} → {current_metric:.4f} "
            f"(drop={relative_drop:.1%}) [{status}]"
        )

        # Return the result
        return result

    def plot_drift_report(
        self,
        drift_report: Dict[str, Dict],
        filename: str = "drift_report.png",
    ) -> str:
        """
        Generate a visual drift report showing PSI for all features.

        Parameters
        ----------
        drift_report : dict
            Output from check_feature_drift().
        filename : str
            Output filename.

        Returns
        -------
        str
            Path to the saved plot.
        """
        # Extract feature names and PSI values
        features = list(drift_report.keys())
        psi_values = [drift_report[f]["psi"] for f in features]

        # Sort by PSI value (highest drift first)
        sorted_indices = np.argsort(psi_values)[::-1]
        features = [features[i] for i in sorted_indices]
        psi_values = [psi_values[i] for i in sorted_indices]

        # Limit to top 20 features for readability
        features = features[:20]
        psi_values = psi_values[:20]

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(12, 8))

        # Color bars by drift severity
        colors = []
        for psi in psi_values:
            if psi >= self.psi_threshold:
                colors.append("red")                   # Significant drift
            elif psi >= 0.1:
                colors.append("orange")                # Warning
            else:
                colors.append("green")                 # Stable

        # Plot horizontal bars
        ax.barh(range(len(features)), psi_values, color=colors, edgecolor="black", alpha=0.8)

        # Add threshold lines
        ax.axvline(x=self.psi_threshold, color="red", linestyle="--",
                    label=f"Drift threshold ({self.psi_threshold})", linewidth=1.5)
        ax.axvline(x=0.1, color="orange", linestyle="--",
                    label="Warning threshold (0.1)", linewidth=1)

        # Set labels
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel("PSI Value", fontsize=12)
        ax.set_title("Feature Drift Report (PSI)", fontsize=14)
        ax.legend(fontsize=10)
        ax.invert_yaxis()                              # Highest PSI at top
        ax.grid(True, alpha=0.3, axis="x")

        # Save the plot
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Log and return
        logger.info(f"Drift report plot saved to: {filepath}")
        return str(filepath)

    def generate_monitoring_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of all monitoring checks performed.

        Returns
        -------
        dict
            Summary with latest status, trends, and recommendations.
        """
        # Build the summary
        summary = {
            "total_checks": len(self.monitoring_history),
            "latest_check": (
                self.monitoring_history[-1] if self.monitoring_history else None
            ),
            "recommendation": "stable",
        }

        # Analyze trend if we have enough history
        if len(self.monitoring_history) >= 2:
            # Compare latest drift count to previous
            latest = self.monitoring_history[-1]["features_drifted"]
            previous = self.monitoring_history[-2]["features_drifted"]

            if latest > previous:
                summary["recommendation"] = "drift_increasing"
                summary["action"] = "Consider retraining - drift is increasing"
            elif latest > 3:
                summary["recommendation"] = "high_drift"
                summary["action"] = "Multiple features drifted - retraining recommended"
            else:
                summary["recommendation"] = "stable"
                summary["action"] = "No action needed"
        else:
            summary["action"] = "Insufficient history for trend analysis"

        # Log the summary
        logger.info(f"Monitoring summary: {summary['recommendation']}")

        # Return the summary
        return summary

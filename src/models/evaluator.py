# ============================================================
# src/models/evaluator.py
# Advanced model evaluation: PR-AUC, lift@k, expected profit
# curves, calibration plots, and comprehensive comparison.
# ============================================================

import numpy as np                                 # NumPy for numerical operations
import pandas as pd                                # Pandas for DataFrames
from sklearn.metrics import (
    roc_auc_score,                                 # Area Under ROC Curve
    average_precision_score,                       # Average Precision (PR-AUC)
    precision_recall_curve,                        # Precision-Recall curve points
    roc_curve,                                     # ROC curve points
    f1_score,                                      # F1 Score
    precision_score,                               # Precision
    recall_score,                                  # Recall (sensitivity)
    confusion_matrix,                              # Confusion matrix
    classification_report,                         # Full classification report
    brier_score_loss,                              # Calibration error (lower=better)
    log_loss,                                      # Log loss (cross-entropy)
)
from sklearn.calibration import calibration_curve  # Calibration plot data
import matplotlib.pyplot as plt                    # Matplotlib for plotting
plt.switch_backend('Agg')                          # Non-interactive backend for server environments
import seaborn as sns                              # Statistical data visualization
from loguru import logger                          # Structured logging
from typing import Dict, Any, List, Optional, Tuple  # Type hints
from pathlib import Path                           # Object-oriented paths


class ModelEvaluator:
    """
    Comprehensive model evaluation for churn prediction.

    Goes beyond simple accuracy to evaluate:
    - PR-AUC (critical for imbalanced datasets)
    - Lift @ k% (how much better than random in top segments)
    - Expected profit curves (business-oriented evaluation)
    - Calibration quality (are probabilities meaningful?)
    - Multi-model comparison
    """

    def __init__(self, config: dict, output_dir: str = "reports"):
        """
        Initialize the evaluator with configuration.

        Parameters
        ----------
        config : dict
            Project configuration dictionary.
        output_dir : str
            Directory where evaluation plots and reports will be saved.
        """
        # Store the configuration
        self.config = config
        # Set up the output directory for reports and plots
        self.output_dir = Path(output_dir)
        # Create the output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Extract the top-k percentiles for lift analysis
        self.top_k = config["evaluation"]["top_k_percentiles"]
        # Extract threshold optimization settings
        self.retention_cost = config["model"]["threshold"]["retention_cost"]
        self.customer_value = config["model"]["threshold"]["avg_customer_value"]
        # Log initialization
        logger.info(f"ModelEvaluator initialized. Output dir: {self.output_dir}")

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
        model_name: str = "model",
    ) -> Dict[str, float]:
        """
        Compute a comprehensive set of classification metrics.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels (0 or 1).
        y_proba : np.ndarray
            Predicted churn probabilities (0.0 to 1.0).
        threshold : float
            Decision threshold for converting probabilities to labels.
        model_name : str
            Name of the model (for logging).

        Returns
        -------
        dict
            Dictionary mapping metric names to their values.
        """
        # Support two call signatures for convenience/tests:
        # 1) (y_true, y_proba, threshold=float, model_name=str)
        # 2) (model, X, y, model_name=str) -- when a model object is passed first
        if hasattr(y_true, "predict_proba"):
            # Called as (model, X, y, ...)
            model = y_true
            X = y_proba
            y = threshold
            # Compute probabilities for the positive class
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                y_proba = 1 / (1 + np.exp(-scores))
            else:
                raise ValueError("Model has neither predict_proba nor decision_function")
            y_true = np.asarray(y)
            # Use provided model_name if available
            model_name = getattr(model, "__class__", model_name)

        # Ensure numpy arrays for downstream operations
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        # Convert probabilities to binary predictions using the threshold
        y_pred = (y_proba >= threshold).astype(int)

        # Compute the confusion matrix: [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Build the comprehensive metrics dictionary
        metrics = {
            "model": model_name,                       # Model identifier
            "threshold": threshold,                    # Decision threshold used
            "roc_auc": round(roc_auc_score(y_true, y_proba), 4),       # ROC-AUC
            "pr_auc": round(average_precision_score(y_true, y_proba), 4),  # PR-AUC
            "f1": round(f1_score(y_true, y_pred), 4),                  # F1 Score
            "precision": round(precision_score(y_true, y_pred), 4),    # Precision
            "recall": round(recall_score(y_true, y_pred), 4),         # Recall
            "brier_score": round(brier_score_loss(y_true, y_proba), 4), # Calibration error
            "log_loss": round(log_loss(y_true, y_proba), 4),          # Cross-entropy loss
            "true_positives": int(tp),                 # Correctly identified churners
            "false_positives": int(fp),                # Non-churners flagged as churn
            "true_negatives": int(tn),                 # Correctly identified non-churners
            "false_negatives": int(fn),                # Churners missed by the model
            "specificity": round(tn / (tn + fp), 4) if (tn + fp) > 0 else 0,  # True negative rate
        }

        # Log the key metrics
        logger.info(f"Metrics for '{model_name}' (threshold={threshold}):")
        logger.info(f"  ROC-AUC: {metrics['roc_auc']}")
        logger.info(f"  PR-AUC:  {metrics['pr_auc']}")
        logger.info(f"  F1:      {metrics['f1']}")
        logger.info(f"  Recall:  {metrics['recall']}")
        logger.info(f"  Brier:   {metrics['brier_score']}")

        # Return the metrics dictionary
        return metrics

    def compute_lift_at_k(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        k_percentiles: Optional[List[int]] = None,
        k: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Compute lift at various top-k percentiles.

        Lift measures how much better the model is than random selection.
        A lift of 3.0 at top 10% means the top 10% of predictions
        catches 3x more churners than a random 10% sample.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels.
        y_proba : np.ndarray
            Predicted churn probabilities.
        k_percentiles : list of int, optional
            Percentiles to compute lift for (e.g., [5, 10, 20]).

        Returns
        -------
        dict
            Dictionary mapping 'lift_at_{k}%' to lift values.
        """
        # Support call signature where a model is supplied first: (model, X, y, k=0.1)
        if hasattr(y_true, "predict_proba"):
            model = y_true
            X = y_proba
            y = k_percentiles
            # Compute probabilities
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                y_proba = 1 / (1 + np.exp(-scores))
            else:
                raise ValueError("Model has neither predict_proba nor decision_function")
            y_true = np.asarray(y)

        # Use configured percentiles if none provided
        if k_percentiles is None:
            k_percentiles = self.top_k

        # If a single 'k' argument (fraction or percent) was provided, normalize to list
        requested_percent = None
        if k is not None:
            # convert fractional to percentile if necessary
            if k <= 1:
                k_percent = int(k * 100)
            else:
                k_percent = int(k)
            k_percentiles = [k_percent]
            requested_percent = k_percent

        # Overall churn rate (baseline for random selection)
        base_rate = np.mean(y_true)

        # Sort customers by predicted churn probability (highest first)
        sorted_indices = np.argsort(-y_proba)          # Descending sort indices
        n_total = len(y_true)                          # Total number of customers

        # Initialize the lift results dictionary
        lift_results = {}

        # Compute lift at each percentile
        for k in k_percentiles:
            # Number of customers in the top k%
            n_top_k = max(1, int(n_total * k / 100))
            # Get the indices of the top k% predictions
            top_k_indices = sorted_indices[:n_top_k]
            # Churn rate in the top k% segment
            top_k_churn_rate = np.mean(y_true[top_k_indices])
            # Lift = segment churn rate / overall churn rate
            lift = top_k_churn_rate / base_rate if base_rate > 0 else 0
            # Store the result
            lift_results[f"lift_at_{k}%"] = round(lift, 2)
            # Also compute recall at k (what fraction of all churners are in top k%)
            recall_at_k = np.sum(y_true[top_k_indices]) / np.sum(y_true) if np.sum(y_true) > 0 else 0
            lift_results[f"recall_at_{k}%"] = round(recall_at_k, 4)

            # Log the lift results
            logger.info(f"  Top {k}%: Lift={lift:.2f}x, Recall={recall_at_k:.1%}")

        # If a single k was requested, return the numeric lift for that percentile
        if requested_percent is not None and len(k_percentiles) == 1:
            return lift_results.get(f"lift_at_{requested_percent}%", 0)

        # Otherwise return the full dictionary of lifts and recalls
        return lift_results

    def compute_expected_profit(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute expected profit at various thresholds.

        This is the key business metric: how much money do we save
        by targeting the right customers for retention?

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels.
        y_proba : np.ndarray
            Predicted churn probabilities.

        Returns
        -------
        dict
            Dictionary with optimal threshold, max profit, and profit curve data.
        """
        # Generate a range of thresholds to evaluate
        thresholds = np.arange(0.1, 0.91, 0.01)
        # List to store profit at each threshold
        profits = []

        # Evaluate profit at each threshold
        for t in thresholds:
            # Classify as churn risk if probability >= threshold
            predicted = (y_proba >= t).astype(int)
            # True positives (churners correctly targeted)
            tp = np.sum((predicted == 1) & (y_true == 1))
            # False positives (non-churners incorrectly targeted)
            fp = np.sum((predicted == 1) & (y_true == 0))
            # Total targeted customers
            total_targeted = tp + fp
            # Revenue saved (assume 50% of targeted churners are actually saved)
            revenue_saved = tp * self.customer_value * 0.5
            # Cost of retention offers
            cost = total_targeted * self.retention_cost
            # Net profit
            profit = revenue_saved - cost
            # Append to the list
            profits.append(profit)

        # Convert to numpy array for indexing
        profits = np.array(profits)

        # Find the threshold that maximizes profit
        best_idx = np.argmax(profits)
        best_threshold = round(float(thresholds[best_idx]), 2)
        best_profit = round(float(profits[best_idx]), 2)

        # Build the result dictionary
        result = {
            "optimal_threshold": best_threshold,       # Best decision threshold
            "max_profit": best_profit,                 # Maximum expected profit
            "thresholds": thresholds.tolist(),          # All evaluated thresholds
            "profits": profits.tolist(),                # Profit at each threshold
            "retention_cost": self.retention_cost,     # Cost per retention offer
            "customer_value": self.customer_value,     # Average customer revenue
        }

        # Log the result
        logger.info(f"Expected profit analysis:")
        logger.info(f"  Optimal threshold: {best_threshold}")
        logger.info(f"  Maximum profit: ${best_profit:,.0f}")

        # Return the profit analysis
        return result

    def plot_roc_curve(
        self,
        models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        filename: str = "roc_curves.png",
    ) -> str:
        """
        Plot ROC curves for multiple models on the same axes.

        Parameters
        ----------
        models_data : dict
            Dictionary mapping model names to (y_true, y_proba) tuples.
        filename : str
            Output filename for the plot.

        Returns
        -------
        str
            Path to the saved plot file.
        """
        # Create a figure with specified size
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot ROC curve for each model
        for name, (y_true, y_proba) in models_data.items():
            # Compute the ROC curve points
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            # Compute the AUC score
            auc = roc_auc_score(y_true, y_proba)
            # Plot the curve with the AUC in the legend
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

        # Plot the diagonal (random classifier baseline)
        ax.plot([0, 1], [0, 1], "k--", label="Random (AUC=0.500)", linewidth=1)

        # Set axis labels and title
        ax.set_xlabel("False Positive Rate", fontsize=12)   # X-axis label
        ax.set_ylabel("True Positive Rate", fontsize=12)    # Y-axis label
        ax.set_title("ROC Curve Comparison", fontsize=14)   # Plot title
        ax.legend(loc="lower right", fontsize=10)            # Legend position
        ax.grid(True, alpha=0.3)                             # Light grid for readability

        # Save the plot to disk
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        # Close the figure to free memory
        plt.close(fig)

        # Log the save location
        logger.info(f"ROC curve plot saved to: {filepath}")

        # Return the filepath as string
        return str(filepath)

    def plot_precision_recall_curve(
        self,
        models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        filename: str = "pr_curves.png",
    ) -> str:
        """
        Plot Precision-Recall curves for multiple models.

        PR curves are more informative than ROC for imbalanced datasets
        because they focus on the minority (churn) class.

        Parameters
        ----------
        models_data : dict
            Dictionary mapping model names to (y_true, y_proba) tuples.
        filename : str
            Output filename for the plot.

        Returns
        -------
        str
            Path to the saved plot file.
        """
        # Create a figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot PR curve for each model
        for name, (y_true, y_proba) in models_data.items():
            # Compute precision-recall curve points
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            # Compute average precision (area under PR curve)
            ap = average_precision_score(y_true, y_proba)
            # Plot the curve
            ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})", linewidth=2)

        # Plot the baseline (random classifier = overall churn rate)
        # Get churn rate from first model's data
        first_y = list(models_data.values())[0][0]
        baseline = np.mean(first_y)
        ax.axhline(y=baseline, color="k", linestyle="--",
                    label=f"Baseline ({baseline:.3f})", linewidth=1)

        # Set axis labels and title
        ax.set_xlabel("Recall", fontsize=12)                 # X-axis
        ax.set_ylabel("Precision", fontsize=12)              # Y-axis
        ax.set_title("Precision-Recall Curve Comparison", fontsize=14)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        # Save the plot
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Log and return
        logger.info(f"PR curve plot saved to: {filepath}")
        return str(filepath)

    def plot_calibration_curve(
        self,
        models_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
        filename: str = "calibration_curves.png",
        n_bins: int = 10,
    ) -> str:
        """
        Plot calibration curves to assess probability quality.

        A well-calibrated model's curve follows the diagonal:
        predicted probability matches observed frequency.

        Parameters
        ----------
        models_data : dict
            Dictionary mapping model names to (y_true, y_proba) tuples.
        filename : str
            Output filename.
        n_bins : int
            Number of bins for calibration curve.

        Returns
        -------
        str
            Path to the saved plot.
        """
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Plot calibration curves
        for name, (y_true, y_proba) in models_data.items():
            # Compute calibration curve (fraction of positives vs mean predicted)
            fraction_of_positives, mean_predicted = calibration_curve(
                y_true, y_proba, n_bins=n_bins, strategy="uniform"
            )
            # Compute Brier score (lower = better calibration)
            brier = brier_score_loss(y_true, y_proba)
            # Plot on the left subplot
            ax1.plot(
                mean_predicted, fraction_of_positives,
                marker="o", label=f"{name} (Brier={brier:.3f})", linewidth=2
            )

            # Plot histogram of predicted probabilities on the right subplot
            ax2.hist(y_proba, bins=50, alpha=0.5, label=name, density=True)

        # Plot perfect calibration diagonal
        ax1.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=1)

        # Left subplot settings
        ax1.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax1.set_ylabel("Fraction of Positives", fontsize=12)
        ax1.set_title("Calibration Curve", fontsize=14)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)

        # Right subplot settings
        ax2.set_xlabel("Predicted Probability", fontsize=12)
        ax2.set_ylabel("Density", fontsize=12)
        ax2.set_title("Prediction Distribution", fontsize=14)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Adjust layout
        fig.tight_layout()

        # Save the plot
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Log and return
        logger.info(f"Calibration plot saved to: {filepath}")
        return str(filepath)

    def plot_profit_curve(
        self,
        profit_data: Dict[str, Any],
        filename: str = "profit_curve.png",
    ) -> str:
        """
        Plot the expected profit as a function of the decision threshold.

        Parameters
        ----------
        profit_data : dict
            Output from compute_expected_profit().
        filename : str
            Output filename.

        Returns
        -------
        str
            Path to the saved plot.
        """
        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the profit curve
        ax.plot(
            profit_data["thresholds"],                 # X: thresholds
            profit_data["profits"],                    # Y: profits
            linewidth=2,
            color="green",
            label="Expected Profit",
        )

        # Mark the optimal threshold
        ax.axvline(
            x=profit_data["optimal_threshold"],
            color="red", linestyle="--", linewidth=1.5,
            label=f"Optimal threshold ({profit_data['optimal_threshold']})",
        )

        # Mark the zero profit line
        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)

        # Set labels and title
        ax.set_xlabel("Decision Threshold", fontsize=12)
        ax.set_ylabel("Expected Profit ($)", fontsize=12)
        ax.set_title(
            f"Expected Profit Curve (Max: ${profit_data['max_profit']:,.0f})",
            fontsize=14,
        )
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Save the plot
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)

        # Log and return
        logger.info(f"Profit curve saved to: {filepath}")
        return str(filepath)

    def compare_models(
        self,
        models_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> pd.DataFrame:
        """
        Compare all models using a comprehensive metrics table.

        Parameters
        ----------
        models_predictions : dict
            Dictionary mapping model names to (y_true, y_proba) tuples.

        Returns
        -------
        pd.DataFrame
            Comparison DataFrame with all metrics for each model.
        """
        # Initialize list to collect metrics for each model
        all_metrics = []

        # Log the comparison
        logger.info("=" * 60)
        logger.info("Model Comparison")
        logger.info("=" * 60)

        # Compute metrics for each model
        for name, (y_true, y_proba) in models_predictions.items():
            # Get classification metrics
            metrics = self.compute_metrics(y_true, y_proba, model_name=name)
            # Get lift metrics
            lift = self.compute_lift_at_k(y_true, y_proba)
            # Get profit metrics
            profit = self.compute_expected_profit(y_true, y_proba)
            # Merge all metrics into one dictionary
            metrics.update(lift)
            metrics["optimal_threshold"] = profit["optimal_threshold"]
            metrics["max_profit"] = profit["max_profit"]
            # Append to the list
            all_metrics.append(metrics)

        # Create comparison DataFrame
        comparison_df = pd.DataFrame(all_metrics)

        # Sort by PR-AUC (most important metric for imbalanced data)
        comparison_df = comparison_df.sort_values("pr_auc", ascending=False)

        # Reset index for clean display
        comparison_df = comparison_df.reset_index(drop=True)

        # Log the comparison table
        logger.info("\nModel Comparison Summary:")
        logger.info(f"\n{comparison_df.to_string()}")

        # Generate all comparison plots
        self.plot_roc_curve(models_predictions)
        self.plot_precision_recall_curve(models_predictions)
        self.plot_calibration_curve(models_predictions)

        # Return the comparison DataFrame
        return comparison_df

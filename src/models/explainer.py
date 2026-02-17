# ============================================================
# src/models/explainer.py
# Model explainability: SHAP values for global and local
# explanations, what-if analysis, and counterfactual recourse.
# ============================================================

import numpy as np                                 # NumPy for numerical operations
import pandas as pd                                # Pandas for DataFrames
import shap                                        # SHAP (SHapley Additive exPlanations)
import matplotlib.pyplot as plt                    # Matplotlib for plotting
plt.switch_backend('Agg')                          # Non-interactive backend for server environments
from loguru import logger                          # Structured logging
from typing import Dict, Any, List, Optional       # Type hints
from pathlib import Path                           # Object-oriented paths


class ModelExplainer:
    """
    Provides model explainability using SHAP values.

    Capabilities:
    1. Global explanations: which features matter most overall
    2. Local explanations: why a specific customer is flagged
    3. What-if analysis: how changing a feature changes the prediction
    4. Counterfactual recourse: minimal changes to flip a prediction
    """

    def __init__(
        self,
        model: Any,
        X_train: pd.DataFrame,
        feature_names: List[str],
        output_dir: str = "reports",
    ):
        """
        Initialize the explainer with a trained model and training data.

        Parameters
        ----------
        model : object
            The trained model (must support predict_proba).
        X_train : pd.DataFrame or np.ndarray
            Training data used to build the SHAP background dataset.
        feature_names : list of str
            Names of the features (for human-readable explanations).
        output_dir : str
            Directory for saving explanation plots.
        """
        # Store the trained model
        self.model = model
        # Store the feature names for readable output
        self.feature_names = feature_names
        # Set up output directory for plots
        self.output_dir = Path(output_dir)
        # Create the directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Convert X_train to DataFrame if it's a numpy array
        if isinstance(X_train, np.ndarray):
            # Create DataFrame with proper column names
            self.X_train = pd.DataFrame(X_train, columns=feature_names)
        else:
            # Already a DataFrame, just store it
            self.X_train = X_train

        # Create a SHAP background dataset (subsample for speed)
        # Using 100 samples is usually sufficient for accurate SHAP values
        n_background = min(100, len(self.X_train))
        self.background = shap.sample(self.X_train, n_background)

        # Initialize the SHAP explainer
        # TreeExplainer is fastest for tree-based models
        try:
            # Try tree-based explainer first (works for XGBoost, LightGBM, RF)
            self.explainer = shap.TreeExplainer(model)
            self.explainer_type = "tree"
            logger.info("Using SHAP TreeExplainer (optimized for tree models)")
        except Exception:
            # Fall back to KernelExplainer (works for any model)
            self.explainer = shap.KernelExplainer(
                model.predict_proba,                   # The model's prediction function
                self.background,                       # Background dataset for integration
            )
            self.explainer_type = "kernel"
            logger.info("Using SHAP KernelExplainer (model-agnostic)")

        # Store computed SHAP values for reuse
        self.shap_values = None
        # Log initialization
        logger.info(f"ModelExplainer initialized with {len(feature_names)} features")

    def compute_shap_values(
        self,
        X: pd.DataFrame,
        max_samples: int = 500,
    ) -> np.ndarray:
        """
        Compute SHAP values for a set of samples.

        SHAP values decompose each prediction into per-feature contributions.
        Positive SHAP = pushes prediction toward churn.
        Negative SHAP = pushes prediction away from churn.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Samples to explain.
        max_samples : int
            Maximum number of samples to compute SHAP for (performance).

        Returns
        -------
        np.ndarray
            SHAP values array of shape (n_samples, n_features).
        """
        # Convert to DataFrame if needed
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)

        # Subsample if too many samples (SHAP can be slow)
        if len(X) > max_samples:
            logger.info(f"Subsampling from {len(X)} to {max_samples} for SHAP computation")
            X = X.sample(n=max_samples, random_state=42)

        # Log the computation start
        logger.info(f"Computing SHAP values for {len(X)} samples...")

        # Compute SHAP values
        shap_values = self.explainer.shap_values(X)

        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # For binary classification, take the positive class (index 1)
            shap_values = shap_values[1]

        # Store for reuse
        self.shap_values = shap_values
        self._shap_X = X

        # Log completion
        logger.info(f"SHAP values computed. Shape: {shap_values.shape}")

        # Return the SHAP values array
        return shap_values

    def get_global_importance(
        self,
        shap_values: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """
        Get global feature importance based on mean absolute SHAP values.

        This shows which features matter most across all customers.

        Parameters
        ----------
        shap_values : np.ndarray, optional
            Pre-computed SHAP values. If None, uses stored values.

        Returns
        -------
        pd.DataFrame
            DataFrame with features ranked by importance.
        """
        # Use stored SHAP values if none provided
        if shap_values is None:
            shap_values = self.shap_values

        # Check that SHAP values are available
        if shap_values is None:
            raise RuntimeError("No SHAP values available. Call compute_shap_values() first.")

        # Compute mean absolute SHAP value for each feature
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        # Create a DataFrame with feature names and importance
        importance_df = pd.DataFrame({
            "feature": self.feature_names,             # Feature names
            "mean_abs_shap": mean_abs_shap,           # Mean |SHAP| value
        })

        # Sort by importance (descending)
        importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
        # Reset index
        importance_df = importance_df.reset_index(drop=True)

        # Log top features
        logger.info("Global feature importance (top 10):")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")

        # Return the importance DataFrame
        return importance_df

    def explain_customer(
        self,
        customer_data: pd.DataFrame,
        customer_index: int = 0,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """
        Generate a local explanation for a specific customer.

        Shows why the model predicted this customer's churn probability,
        listing the top contributing factors.

        Parameters
        ----------
        customer_data : pd.DataFrame or np.ndarray
            Feature data for the customer(s) to explain.
        customer_index : int
            Index of the customer to explain (if multiple rows).
        top_n : int
            Number of top contributing features to return.

        Returns
        -------
        dict
            Dictionary with prediction, top positive drivers,
            top negative drivers, and all feature contributions.
        """
        # Convert to DataFrame if needed
        if isinstance(customer_data, np.ndarray):
            customer_data = pd.DataFrame(customer_data, columns=self.feature_names)

        # Get the single customer's data
        if len(customer_data) > 1:
            customer_row = customer_data.iloc[[customer_index]]
        else:
            customer_row = customer_data

        # Get the model's prediction for this customer
        churn_probability = self.model.predict_proba(customer_row)[0][1]

        # Compute SHAP values for this customer
        customer_shap = self.explainer.shap_values(customer_row)

        # Handle different SHAP output formats
        if isinstance(customer_shap, list):
            customer_shap = customer_shap[1]

        # Flatten to 1D array
        shap_vals = customer_shap.flatten()

        # Create a DataFrame of feature contributions
        contributions = pd.DataFrame({
            "feature": self.feature_names,             # Feature names
            "value": customer_row.values.flatten(),   # Feature values
            "shap_value": shap_vals,                  # SHAP contributions
            "abs_shap": np.abs(shap_vals),            # Absolute SHAP (for sorting)
        })

        # Sort by absolute contribution (most impactful first)
        contributions = contributions.sort_values("abs_shap", ascending=False)

        # Split into positive (pushing toward churn) and negative (away from churn)
        positive_drivers = contributions[contributions["shap_value"] > 0].head(top_n)
        negative_drivers = contributions[contributions["shap_value"] < 0].head(top_n)

        # Build the explanation dictionary
        explanation = {
            "churn_probability": round(float(churn_probability), 4),
            "risk_level": (
                "HIGH" if churn_probability >= 0.7 else          # 70%+ = high risk
                "MEDIUM" if churn_probability >= 0.4 else        # 40-70% = medium risk
                "LOW"                                            # <40% = low risk
            ),
            "top_churn_drivers": [                     # Factors pushing toward churn
                {
                    "feature": row["feature"],
                    "value": row["value"],
                    "impact": round(float(row["shap_value"]), 4),
                }
                for _, row in positive_drivers.iterrows()
            ],
            "top_retention_factors": [                 # Factors pushing away from churn
                {
                    "feature": row["feature"],
                    "value": row["value"],
                    "impact": round(float(row["shap_value"]), 4),
                }
                for _, row in negative_drivers.iterrows()
            ],
            "all_contributions": contributions.to_dict("records"),
        }

        # Log the explanation summary
        logger.info(f"Customer explanation: P(churn)={churn_probability:.1%}, Risk={explanation['risk_level']}")

        # Return the explanation
        return explanation

    def what_if_analysis(
        self,
        customer_data: pd.DataFrame,
        feature_changes: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform what-if analysis: how does changing features affect the prediction?

        Example: "What if this customer switches from month-to-month to yearly?"

        Parameters
        ----------
        customer_data : pd.DataFrame
            Original customer feature data.
        feature_changes : dict
            Dictionary of feature_name: new_value pairs.

        Returns
        -------
        dict
            Dictionary with original and new predictions and the delta.
        """
        # Convert to DataFrame if needed
        if isinstance(customer_data, np.ndarray):
            customer_data = pd.DataFrame(customer_data, columns=self.feature_names)

        # Get original prediction
        original_prob = float(self.model.predict_proba(customer_data)[0][1])

        # Create a modified copy with the proposed changes
        modified_data = customer_data.copy()
        for feature, new_value in feature_changes.items():
            # Check that the feature exists
            if feature in modified_data.columns:
                # Apply the change
                modified_data[feature] = new_value
                logger.info(f"  Changed '{feature}' to {new_value}")
            else:
                # Log a warning if the feature doesn't exist
                logger.warning(f"  Feature '{feature}' not found — skipping")

        # Get the new prediction after changes
        new_prob = float(self.model.predict_proba(modified_data)[0][1])

        # Calculate the change in churn probability
        delta = new_prob - original_prob

        # Build the result
        result = {
            "original_probability": round(original_prob, 4),   # Before changes
            "new_probability": round(new_prob, 4),             # After changes
            "probability_change": round(delta, 4),             # Difference
            "percentage_change": round(delta / original_prob * 100, 1) if original_prob > 0 else 0,
            "changes_applied": feature_changes,                # What was changed
            "risk_reduced": delta < 0,                         # Did risk decrease?
        }

        # Log the what-if result
        logger.info(f"What-if analysis: {original_prob:.1%} → {new_prob:.1%} (Δ={delta:+.1%})")

        # Return the result
        return result

    def plot_global_summary(
        self,
        shap_values: Optional[np.ndarray] = None,
        X: Optional[pd.DataFrame] = None,
        filename: str = "shap_summary.png",
        max_display: int = 20,
    ) -> str:
        """
        Generate and save a SHAP summary plot (beeswarm plot).

        This shows global feature importance AND the direction of effects.

        Parameters
        ----------
        shap_values : np.ndarray, optional
            SHAP values. If None, uses stored values.
        X : pd.DataFrame, optional
            Feature values for coloring. If None, uses stored data.
        filename : str
            Output filename.
        max_display : int
            Maximum features to show.

        Returns
        -------
        str
            Path to the saved plot.
        """
        # Use stored values if not provided
        if shap_values is None:
            shap_values = self.shap_values
        if X is None:
            X = self._shap_X

        # Check SHAP values are available
        if shap_values is None:
            raise RuntimeError("No SHAP values available.")

        # Create the SHAP summary plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values,                               # SHAP values matrix
            X,                                         # Feature values (for coloring)
            feature_names=self.feature_names,          # Feature labels
            max_display=max_display,                   # Max features to show
            show=False,                                # Don't display inline
        )

        # Save the plot
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close("all")

        # Log and return
        logger.info(f"SHAP summary plot saved to: {filepath}")
        return str(filepath)

    def plot_customer_waterfall(
        self,
        customer_data: pd.DataFrame,
        customer_index: int = 0,
        filename: str = "shap_waterfall.png",
    ) -> str:
        """
        Generate a waterfall plot showing how each feature
        contributes to a specific customer's prediction.

        Parameters
        ----------
        customer_data : pd.DataFrame
            Feature data for the customer to explain.
        customer_index : int
            Index of the customer in the data.
        filename : str
            Output filename.

        Returns
        -------
        str
            Path to the saved plot.
        """
        # Convert to DataFrame if needed
        if isinstance(customer_data, np.ndarray):
            customer_data = pd.DataFrame(customer_data, columns=self.feature_names)

        # Get single customer
        customer_row = customer_data.iloc[[customer_index]]

        # Compute SHAP values
        shap_vals = self.explainer.shap_values(customer_row)

        # Handle list output
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]

        # Get the expected value (base rate)
        if isinstance(self.explainer.expected_value, (list, np.ndarray)):
            expected = self.explainer.expected_value[1]
        else:
            expected = self.explainer.expected_value

        # Create SHAP Explanation object for waterfall plot
        shap_explanation = shap.Explanation(
            values=shap_vals.flatten(),                # SHAP values for this customer
            base_values=expected,                      # Base prediction (average)
            data=customer_row.values.flatten(),       # Feature values
            feature_names=self.feature_names,          # Feature names
        )

        # Create the waterfall plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.waterfall_plot(shap_explanation, max_display=15, show=False)

        # Save the plot
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close("all")

        # Log and return
        logger.info(f"SHAP waterfall plot saved to: {filepath}")
        return str(filepath)

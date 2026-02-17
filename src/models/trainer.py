# ============================================================
# src/models/trainer.py
# Model training module: trains multiple algorithms, performs
# probability calibration, and optimizes decision thresholds.
# ============================================================

import numpy as np                                 # NumPy for numerical operations
import pandas as pd                                # Pandas for DataFrames
from sklearn.linear_model import LogisticRegression  # Linear baseline model
from sklearn.ensemble import RandomForestClassifier  # Ensemble tree model
from sklearn.calibration import CalibratedClassifierCV  # Probability calibration wrapper
from sklearn.model_selection import cross_val_score  # Cross-validation scoring
import xgboost as xgb                              # XGBoost gradient boosting
import lightgbm as lgb                             # LightGBM gradient boosting
from loguru import logger                          # Structured logging
from typing import Dict, Any, Optional, Tuple      # Type hints


class ModelTrainer:
    """
    Trains multiple classification models for churn prediction.

    Key capabilities:
    - Trains Logistic Regression, Random Forest, XGBoost, LightGBM
    - Applies probability calibration (Isotonic or Platt scaling)
    - Optimizes decision thresholds for business value
    - Supports cost-sensitive learning through class weights
    """

    def __init__(self, config: dict):
        """
        Initialize the ModelTrainer with configuration settings.

        Parameters
        ----------
        config : dict
            Project configuration with model hyperparameters.
        """
        # Store the full configuration
        self.config = config
        # Extract the list of algorithms to train
        self.algorithms = config["model"]["algorithms"]
        # Extract the calibration method (isotonic or sigmoid)
        self.calibration_method = config["model"]["calibration_method"]
        # Extract the random state for reproducibility
        self.random_state = config["data"]["random_state"]
        # Dictionary to store all trained models
        self.models: Dict[str, Any] = {}
        # Dictionary to store calibrated versions of models
        self.calibrated_models: Dict[str, Any] = {}
        # Store the best model after comparison
        self.best_model = None
        # Store the name of the best model
        self.best_model_name = None
        # Log initialization
        logger.info(f"ModelTrainer initialized. Algorithms: {self.algorithms}")

    def _build_logistic_regression(self) -> LogisticRegression:
        """
        Build a Logistic Regression model with config hyperparameters.

        Returns
        -------
        LogisticRegression
            Unfitted Logistic Regression classifier.
        """
        # Extract logistic regression params from config
        params = self.config["model"]["logistic_regression_params"]
        # Create the model with the configured hyperparameters
        model = LogisticRegression(
            C=params["C"],                             # Inverse regularization strength
            penalty=params["penalty"],                 # Type of regularization (l2)
            solver=params["solver"],                   # Optimization algorithm
            max_iter=params["max_iter"],               # Maximum iterations for convergence
            class_weight=params["class_weight"],       # Handle class imbalance
            random_state=params["random_state"],       # Seed for reproducibility
        )
        # Log the model configuration
        logger.info(f"Built LogisticRegression with params: {params}")
        # Return the unfitted model
        return model

    def _build_random_forest(self) -> RandomForestClassifier:
        """
        Build a Random Forest model with config hyperparameters.

        Returns
        -------
        RandomForestClassifier
            Unfitted Random Forest classifier.
        """
        # Extract random forest params from config
        params = self.config["model"]["random_forest_params"]
        # Create the model
        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],        # Number of trees
            max_depth=params["max_depth"],              # Max depth per tree
            min_samples_split=params["min_samples_split"],  # Min samples to split
            min_samples_leaf=params["min_samples_leaf"],    # Min samples per leaf
            class_weight=params["class_weight"],       # Handle imbalance
            random_state=params["random_state"],       # Reproducibility
            n_jobs=-1,                                 # Use all CPU cores
        )
        # Log the model configuration
        logger.info(f"Built RandomForest with params: {params}")
        # Return the unfitted model
        return model

    def _build_xgboost(self) -> xgb.XGBClassifier:
        """
        Build an XGBoost model with config hyperparameters.

        Returns
        -------
        xgb.XGBClassifier
            Unfitted XGBoost classifier.
        """
        # Extract XGBoost params from config
        params = self.config["model"]["xgboost_params"]
        # Create the model
        model = xgb.XGBClassifier(
            n_estimators=params["n_estimators"],       # Number of boosting rounds
            max_depth=params["max_depth"],              # Max tree depth
            learning_rate=params["learning_rate"],     # Step size shrinkage
            subsample=params["subsample"],             # Row sampling fraction
            colsample_bytree=params["colsample_bytree"],  # Column sampling fraction
            scale_pos_weight=params["scale_pos_weight"],   # Positive class weight
            eval_metric=params["eval_metric"],         # Evaluation metric
            random_state=params["random_state"],       # Reproducibility
            use_label_encoder=False,                   # Suppress deprecation warning
            verbosity=0,                               # Suppress XGBoost output
        )
        # Log the model configuration
        logger.info(f"Built XGBoost with params: {params}")
        # Return the unfitted model
        return model

    def _build_lightgbm(self) -> lgb.LGBMClassifier:
        """
        Build a LightGBM model with config hyperparameters.

        Returns
        -------
        lgb.LGBMClassifier
            Unfitted LightGBM classifier.
        """
        # Extract LightGBM params from config
        params = self.config["model"]["lightgbm_params"]
        # Create the model
        model = lgb.LGBMClassifier(
            n_estimators=params["n_estimators"],        # Number of boosting iterations
            max_depth=params["max_depth"],              # Max depth (-1 = no limit)
            learning_rate=params["learning_rate"],     # Step size shrinkage
            num_leaves=params["num_leaves"],           # Max leaves per tree
            subsample=params["subsample"],             # Bagging fraction
            colsample_bytree=params["colsample_bytree"],  # Feature fraction
            is_unbalance=params["is_unbalance"],       # Auto-handle imbalance
            random_state=params["random_state"],       # Reproducibility
            verbose=-1,                                # Suppress LightGBM output
        )
        # Log the model configuration
        logger.info(f"Built LightGBM with params: {params}")
        # Return the unfitted model
        return model

    def train_model(
        self,
        name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Any:
        """
        Train a single model by name.

        Parameters
        ----------
        name : str
            Algorithm name (must match config: logistic_regression,
            random_forest, xgboost, lightgbm).
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training target vector.

        Returns
        -------
        object
            The fitted model.
        """
        # Log the start of training
        logger.info(f"Training model: {name}")

        # Build the model based on the algorithm name
        if name == "logistic_regression":
            model = self._build_logistic_regression()
        elif name == "random_forest":
            model = self._build_random_forest()
        elif name == "xgboost":
            model = self._build_xgboost()
        elif name == "lightgbm":
            model = self._build_lightgbm()
        else:
            # Unknown algorithm name
            raise ValueError(f"Unknown algorithm: {name}")

        # Fit the model on training data
        model.fit(X_train, y_train)

        # Store the trained model in the models dictionary
        self.models[name] = model

        # Log successful training
        logger.info(f"Model '{name}' trained successfully")

        # Return the fitted model
        return model

    def train_all_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Train all configured algorithms.

        Parameters
        ----------
        X_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training target vector.

        Returns
        -------
        dict
            Dictionary mapping model names to fitted model objects.
        """
        # Log the start of multi-model training
        logger.info("=" * 60)
        logger.info(f"Training {len(self.algorithms)} models...")
        logger.info("=" * 60)

        # Train each algorithm in the configured list
        for algo_name in self.algorithms:
            # Train this specific model
            self.train_model(algo_name, X_train, y_train)

        # Log completion
        logger.info(f"All {len(self.algorithms)} models trained successfully")

        # Return the dictionary of all trained models
        return self.models

    def calibrate_model(
        self,
        name: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Any:
        """
        Apply probability calibration to a trained model.

        Raw model outputs (predict_proba) are often poorly calibrated.
        Calibration ensures that a predicted probability of 0.3 means
        roughly 30% of those customers actually churn.

        Parameters
        ----------
        name : str
            Name of the model to calibrate (must be already trained).
        X_val : np.ndarray
            Validation feature matrix for calibration fitting.
        y_val : np.ndarray
            Validation target vector for calibration fitting.

        Returns
        -------
        object
            The calibrated model wrapper.
        """
        # Check that the model exists
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found. Train it first.")

        # Get the trained model
        base_model = self.models[name]

        # Log the calibration process
        logger.info(f"Calibrating model '{name}' using {self.calibration_method} method")

        # Wrap the model with CalibratedClassifierCV
        calibrated = CalibratedClassifierCV(
            estimator=base_model,                      # The base model to calibrate
            method=self.calibration_method,            # 'isotonic' or 'sigmoid' (Platt)
            cv="prefit",                               # Model is already fitted
        )

        # Fit the calibration on validation data
        # This learns the mapping from raw probabilities to calibrated ones
        calibrated.fit(X_val, y_val)

        # Store the calibrated model
        self.calibrated_models[name] = calibrated

        # Log completion
        logger.info(f"Model '{name}' calibrated successfully")

        # Return the calibrated model
        return calibrated

    def calibrate_all_models(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calibrate all trained models.

        Parameters
        ----------
        X_val : np.ndarray
            Validation feature matrix.
        y_val : np.ndarray
            Validation target vector.

        Returns
        -------
        dict
            Dictionary mapping model names to calibrated models.
        """
        # Log the start of calibration
        logger.info("Calibrating all trained models...")

        # Calibrate each trained model
        for name in self.models:
            self.calibrate_model(name, X_val, y_val)

        # Log completion
        logger.info(f"All {len(self.calibrated_models)} models calibrated")

        # Return the dictionary of calibrated models
        return self.calibrated_models

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        retention_cost: float = 50.0,
        customer_value: float = 500.0,
    ) -> Tuple[float, float]:
        """
        Find the decision threshold that maximizes expected profit.

        Instead of using the default 0.5 threshold, we optimize for
        business value: savings from retaining customers minus the
        cost of retention offers.

        Parameters
        ----------
        y_true : np.ndarray
            True binary labels.
        y_proba : np.ndarray
            Predicted churn probabilities.
        retention_cost : float
            Cost of making a retention offer ($).
        customer_value : float
            Expected revenue from retaining a customer ($).

        Returns
        -------
        tuple of (float, float)
            (optimal_threshold, maximum_expected_profit)
        """
        # Initialize tracking variables for the best threshold
        best_threshold = 0.5                           # Start with default
        best_profit = float("-inf")                    # Initialize to negative infinity

        # Support two call signatures for convenience/tests:
        # 1) (y_true, y_proba, ...)  -- arrays provided
        # 2) (model, X, y, ...)      -- model object provided first
        if hasattr(y_true, "predict_proba"):
            model = y_true
            X = y_proba
            y = retention_cost  # shift args: retention_cost holds y when called as (model, X, y)
            # compute probabilities
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X)
                y_proba = 1 / (1 + np.exp(-scores))
            else:
                raise ValueError("Model has neither predict_proba nor decision_function")
            y_true = np.asarray(y)
            # restore defaults for retention_cost/customer_value
            retention_cost = float(customer_value) if isinstance(customer_value, (int, float)) else 50.0
            customer_value = 500.0

        # Search over a range of thresholds from 0.1 to 0.9
        thresholds = np.arange(0.1, 0.91, 0.01)       # 81 candidate thresholds

        # Evaluate each candidate threshold
        for threshold in thresholds:
            # Classify customers as churn risk if probability > threshold
            predicted_churn = (y_proba >= threshold).astype(int)

            # Calculate true positives (correctly identified churners who got offers)
            true_positives = np.sum((predicted_churn == 1) & (y_true == 1))
            # Calculate false positives (non-churners who got unnecessary offers)
            false_positives = np.sum((predicted_churn == 1) & (y_true == 0))
            # Total customers targeted with retention offers
            total_targeted = true_positives + false_positives

            # Calculate expected profit:
            # Revenue saved = true positives * customer value (assume 50% save rate)
            revenue_saved = true_positives * customer_value * 0.5
            # Cost = everyone targeted gets a retention offer
            cost = total_targeted * retention_cost
            # Net profit = revenue saved minus retention costs
            profit = revenue_saved - cost

            # Update if this threshold gives higher profit
            if profit > best_profit:
                best_profit = profit
                best_threshold = threshold

        # Log the optimal threshold and profit
        logger.info(f"Optimal threshold: {best_threshold:.2f}")
        logger.info(f"Expected profit at optimal threshold: ${best_profit:,.0f}")

        # Backwards-compatible return values:
        # - If called with a model object first (convenience/tests), return only the threshold (float)
        # - If called with (y_true, y_proba), return (threshold, profit)
        if 'model' in locals():
            return round(best_threshold, 2)
        return round(best_threshold, 2), round(best_profit, 2)

    def cross_validate(
        self,
        name: str,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = "roc_auc",
    ) -> Dict[str, float]:
        """
        Run k-fold cross-validation for a model.

        Parameters
        ----------
        name : str
            Name of the model to cross-validate.
        X : np.ndarray
            Feature matrix for cross-validation.
        y : np.ndarray
            Target vector.
        cv : int
            Number of cross-validation folds.
        scoring : str
            Scoring metric (e.g., 'roc_auc', 'average_precision', 'f1').

        Returns
        -------
        dict
            Dictionary with mean and std of the CV scores.
        """
        # Check that the model exists
        if name not in self.models:
            raise ValueError(f"Model '{name}' not found.")

        # Get the model
        model = self.models[name]

        # Log the start of cross-validation
        logger.info(f"Cross-validating '{name}' with {cv} folds, metric: {scoring}")

        # Run cross-validation
        scores = cross_val_score(
            model,                                     # The fitted model
            X,                                         # Feature matrix
            y,                                         # Target vector
            cv=cv,                                     # Number of folds
            scoring=scoring,                           # Metric to evaluate
            n_jobs=-1,                                 # Use all CPU cores
        )

        # Calculate summary statistics
        result = {
            "mean": round(float(np.mean(scores)), 4),  # Average score across folds
            "std": round(float(np.std(scores)), 4),    # Standard deviation of scores
            "scores": [round(float(s), 4) for s in scores],  # Individual fold scores
        }

        # Log the results
        logger.info(f"  {scoring}: {result['mean']:.4f} (+/- {result['std']:.4f})")

        # Return the CV results
        return result

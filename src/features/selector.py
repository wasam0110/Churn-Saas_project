# ============================================================
# src/features/selector.py
# Feature selection module: identifies the most informative
# features and removes redundant or noisy ones.
# ============================================================

import pandas as pd                                # Pandas for DataFrame operations
import numpy as np                                 # NumPy for numerical computations
from sklearn.feature_selection import (
    mutual_info_classif,                           # Mutual information for feature relevance
    SelectKBest,                                   # Select top K features by score
)
from sklearn.ensemble import RandomForestClassifier  # Tree-based importance ranking
from loguru import logger                          # Structured logging
from typing import List, Tuple, Optional           # Type hints


class FeatureSelector:
    """
    Selects the most predictive features using multiple methods:
    1. Correlation-based filtering (remove highly correlated pairs)
    2. Mutual information scoring (non-linear relevance to target)
    3. Tree-based importance ranking (from Random Forest)

    Combining multiple methods makes selection more robust.
    """

    def __init__(self, config: dict):
        """
        Initialize the FeatureSelector with configuration.

        Parameters
        ----------
        config : dict
            Project configuration dictionary from config.yaml.
        """
        # Store the config for reference
        self.config = config
        # Extract the target column name
        self.target = config["features"]["target"]
        # Store the list of selected feature names after fitting
        self.selected_features: Optional[List[str]] = None
        # Store feature importance scores for reporting
        self.importance_scores: Optional[pd.DataFrame] = None
        # Log initialization
        logger.info("FeatureSelector initialized")

    def remove_high_correlation(
        self,
        df: pd.DataFrame,
        threshold: float = 0.95,
    ) -> pd.DataFrame:
        """
        Remove one feature from each pair of highly correlated features.

        When two features are >95% correlated, they carry nearly the
        same information — keeping both adds noise and computation cost.

        Parameters
        ----------
        df : pd.DataFrame
            Feature DataFrame (no target column).
        threshold : float
            Correlation threshold above which one feature is dropped.

        Returns
        -------
        pd.DataFrame
            DataFrame with redundant features removed.
        """
        # Select only numeric columns for correlation analysis
        numeric_df = df.select_dtypes(include=[np.number])

        # Compute the absolute correlation matrix
        # abs() because we care about strong relationships regardless of direction
        corr_matrix = numeric_df.corr().abs()

        # Create a mask for the upper triangle (avoid duplicate pairs)
        # np.triu creates an upper triangle of True values
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find columns where any correlation exceeds the threshold
        to_drop = [
            column                                     # Column name to drop
            for column in upper_triangle.columns       # Iterate over all columns
            if any(upper_triangle[column] > threshold) # Check if any correlation is too high
        ]

        # Log which columns are being dropped
        if to_drop:
            logger.info(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
        else:
            logger.info("No highly correlated feature pairs found")

        # Drop the redundant columns and return
        return df.drop(columns=to_drop, errors="ignore")

    def mutual_information_scores(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_top: int = 20,
    ) -> pd.DataFrame:
        """
        Compute mutual information between each feature and the target.

        Mutual information captures non-linear dependencies that
        correlation cannot detect.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame (numeric features only).
        y : pd.Series
            Binary target variable.
        n_top : int
            Number of top features to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and MI scores, sorted descending.
        """
        # Select only numeric columns for MI computation
        numeric_X = X.select_dtypes(include=[np.number])

        # Fill any NaN values with 0 (MI cannot handle NaN)
        numeric_X = numeric_X.fillna(0)

        # Compute mutual information scores for each feature vs target
        # random_state ensures reproducibility
        mi_scores = mutual_info_classif(
            numeric_X,                                 # Feature matrix
            y,                                         # Target vector
            random_state=42,                           # Reproducibility seed
        )

        # Create a DataFrame with feature names and their MI scores
        mi_df = pd.DataFrame({
            "feature": numeric_X.columns,              # Feature names
            "mi_score": mi_scores,                     # Mutual information scores
        })

        # Sort by MI score in descending order (most informative first)
        mi_df = mi_df.sort_values("mi_score", ascending=False)

        # Reset the index for clean display
        mi_df = mi_df.reset_index(drop=True)

        # Log the top features
        logger.info(f"Top {n_top} features by mutual information:")
        for _, row in mi_df.head(n_top).iterrows():
            # Log each feature and its MI score
            logger.info(f"  {row['feature']}: {row['mi_score']:.4f}")

        # Return the full MI scores DataFrame
        return mi_df

    def tree_based_importance(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_top: int = 20,
    ) -> pd.DataFrame:
        """
        Compute feature importance using a Random Forest classifier.

        Tree-based importance measures how much each feature reduces
        impurity (Gini) across all trees in the forest.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame (numeric features only).
        y : pd.Series
            Binary target variable.
        n_top : int
            Number of top features to return.

        Returns
        -------
        pd.DataFrame
            DataFrame with feature names and importance scores.
        """
        # Select only numeric columns
        numeric_X = X.select_dtypes(include=[np.number])

        # Fill NaN values with 0 (trees can handle this)
        numeric_X = numeric_X.fillna(0)

        # Create and train a Random Forest for importance ranking
        rf = RandomForestClassifier(
            n_estimators=200,                          # 200 trees for stable importance
            max_depth=10,                              # Limit depth to prevent overfitting
            class_weight="balanced",                   # Handle class imbalance
            random_state=42,                           # Reproducibility
            n_jobs=-1,                                 # Use all CPU cores
        )

        # Fit the Random Forest on the data
        rf.fit(numeric_X, y)

        # Extract feature importances from the fitted model
        importances = rf.feature_importances_

        # Create a DataFrame with feature names and importance scores
        importance_df = pd.DataFrame({
            "feature": numeric_X.columns,              # Feature names
            "importance": importances,                 # Gini importance scores
        })

        # Sort by importance in descending order
        importance_df = importance_df.sort_values("importance", ascending=False)

        # Reset index for clean display
        importance_df = importance_df.reset_index(drop=True)

        # Log the top features
        logger.info(f"Top {n_top} features by tree-based importance:")
        for _, row in importance_df.head(n_top).iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")

        # Return the full importance DataFrame
        return importance_df

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "combined",
        n_features: int = 25,
        correlation_threshold: float = 0.95,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select the best features using the specified method.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame.
        y : pd.Series
            Target variable.
        method : str
            Selection method: "mi" (mutual info), "tree" (RF importance),
            or "combined" (intersection of top features from both).
        n_features : int
            Maximum number of features to select.
        correlation_threshold : float
            Threshold for removing correlated features.

        Returns
        -------
        tuple of (pd.DataFrame, list)
            Selected feature DataFrame and list of selected feature names.
        """
        # Log the start of feature selection
        logger.info(f"Feature selection started. Method: {method}, Max features: {n_features}")

        # Step 1: Remove highly correlated features first
        X_filtered = self.remove_high_correlation(X, threshold=correlation_threshold)

        # Step 2: Apply the chosen selection method
        if method == "mi":
            # Use mutual information scores only
            scores = self.mutual_information_scores(X_filtered, y, n_top=n_features)
            # Select the top n_features by MI score
            selected = scores.head(n_features)["feature"].tolist()

        elif method == "tree":
            # Use tree-based importance only
            scores = self.tree_based_importance(X_filtered, y, n_top=n_features)
            # Select the top n_features by importance
            selected = scores.head(n_features)["feature"].tolist()

        elif method == "combined":
            # Use both methods and take the union of top features
            mi_scores = self.mutual_information_scores(X_filtered, y, n_top=n_features)
            tree_scores = self.tree_based_importance(X_filtered, y, n_top=n_features)

            # Get top features from each method
            mi_top = set(mi_scores.head(n_features)["feature"].tolist())
            tree_top = set(tree_scores.head(n_features)["feature"].tolist())

            # Take union (features that appear in either method's top list)
            combined = mi_top | tree_top

            # Rank by average rank across both methods
            # Create a combined scoring DataFrame
            mi_ranks = mi_scores.reset_index(drop=True)
            mi_ranks["mi_rank"] = range(1, len(mi_ranks) + 1)

            tree_ranks = tree_scores.reset_index(drop=True)
            tree_ranks["tree_rank"] = range(1, len(tree_ranks) + 1)

            # Merge ranks on feature name
            merged = mi_ranks[["feature", "mi_rank"]].merge(
                tree_ranks[["feature", "tree_rank"]],
                on="feature",                          # Join on feature name
                how="outer",                           # Keep all features from both methods
            )

            # Fill missing ranks with a high number (feature not in that method's top)
            max_rank = len(merged) + 1
            merged["mi_rank"] = merged["mi_rank"].fillna(max_rank)
            merged["tree_rank"] = merged["tree_rank"].fillna(max_rank)

            # Compute average rank (lower = better)
            merged["avg_rank"] = (merged["mi_rank"] + merged["tree_rank"]) / 2

            # Sort by average rank and select top n
            merged = merged.sort_values("avg_rank")
            selected = merged.head(n_features)["feature"].tolist()

            # Store the merged scores for reporting
            self.importance_scores = merged

        else:
            # Unknown method: raise an error
            raise ValueError(f"Unknown selection method: {method}. Use 'mi', 'tree', or 'combined'.")

        # Filter selected features to only those that exist in the DataFrame
        selected = [f for f in selected if f in X_filtered.columns]

        # Store the selected feature names
        self.selected_features = selected

        # Log the selection results
        logger.info(f"Selected {len(selected)} features: {selected}")

        # Return the list of selected feature names for backward compatibility with tests
        # and provide a separate method when the caller needs the filtered DataFrame.
        return selected

    def select_features_df(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        method: str = "combined",
        n_features: int = 25,
        correlation_threshold: float = 0.95,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Return the filtered DataFrame along with the list of selected feature names.

        This is the same as the original behavior of `select_features` but exposed
        via a clearly-named method so callers can opt-in to receiving both outputs.
        """
        # Reuse select_features logic (it returns list of feature names)
        selected = self.select_features(X, y, method=method, n_features=n_features, correlation_threshold=correlation_threshold)
        X_filtered = self.remove_high_correlation(X, threshold=correlation_threshold)
        return X_filtered[selected], selected

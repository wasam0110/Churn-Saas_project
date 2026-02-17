# ============================================================
# src/data/preprocessor.py
# Data preprocessing: missing value imputation, encoding,
# scaling, and train/test splitting.
# ============================================================

import pandas as pd                                # Pandas for DataFrame manipulation
import numpy as np                                 # NumPy for numerical operations
from sklearn.model_selection import train_test_split  # Stratified train/test split
from sklearn.preprocessing import (
    StandardScaler,                                # Z-score normalization (mean=0, std=1)
    LabelEncoder,                                  # Encode labels as integers
    OneHotEncoder,                                 # One-hot (dummy) encode categoricals
)
from sklearn.impute import SimpleImputer           # Strategy-based missing value imputation
from sklearn.compose import ColumnTransformer      # Apply different transformers to different columns
from sklearn.pipeline import Pipeline              # Chain preprocessing steps sequentially
from loguru import logger                          # Structured logging
from typing import Tuple, Optional, List           # Type hints


class DataPreprocessor:
    """
    Handles all data preprocessing steps:
    - Missing value imputation
    - Categorical encoding (one-hot)
    - Numerical scaling (standardization)
    - Train/test splitting with stratification

    The preprocessor fits on training data and transforms both
    train and test sets to prevent data leakage.
    """

    def __init__(self, config: dict):
        """
        Initialize the preprocessor with configuration settings.

        Parameters
        ----------
        config : dict
            Project configuration dictionary from config.yaml.
        """
        # Store the full config for reference
        self.config = config
        # Extract numerical column names from config
        self.numerical_cols = config["features"]["numerical_columns"]
        # Extract categorical column names from config
        self.categorical_cols = config["features"]["categorical_columns"]
        # Extract the target column name
        self.target = config["features"]["target"]
        # Extract the test split fraction (e.g., 0.2 = 20%)
        self.test_size = config["data"]["test_size"]
        # Extract the random state seed for reproducibility
        self.random_state = config["data"]["random_state"]

        # Initialize the preprocessing pipeline (will be built in fit())
        self.preprocessor = None
        # Track whether the preprocessor has been fitted
        self.is_fitted = False
        # Store the feature names after transformation (for interpretability)
        self.feature_names = None

        # Log initialization
        logger.info("DataPreprocessor initialized")
        logger.info(f"  Numerical columns ({len(self.numerical_cols)}): {self.numerical_cols}")
        logger.info(f"  Categorical columns ({len(self.categorical_cols)}): {self.categorical_cols}")

    def _build_preprocessor(self) -> ColumnTransformer:
        """
        Build the sklearn ColumnTransformer pipeline that handles
        both numerical and categorical columns.

        Returns
        -------
        ColumnTransformer
            A fitted or unfitted ColumnTransformer with two sub-pipelines.
        """
        # --- Numerical Pipeline ---
        # Step 1: Impute missing values with the median (robust to outliers)
        # Step 2: Standardize to zero mean and unit variance
        numerical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),   # Fill NaN with column median
            ("scaler", StandardScaler()),                     # Scale to mean=0, std=1
        ])

        # --- Categorical Pipeline ---
        # Step 1: Impute missing values with the most frequent category
        # Step 2: One-hot encode to create binary dummy variables
        categorical_pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill NaN with mode
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",              # Ignore unseen categories at predict time
                sparse_output=False,                  # Return dense array (not sparse matrix)
                drop="if_binary",                     # Drop one column for binary features (avoid multicollinearity)
            )),
        ])

        # --- Combine Pipelines ---
        # ColumnTransformer applies each pipeline to its specified columns
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numerical_pipeline, self.numerical_cols),    # Apply numerical pipeline
                ("cat", categorical_pipeline, self.categorical_cols), # Apply categorical pipeline
            ],
            remainder="drop",                        # Drop any columns not in either list
            verbose_feature_names_out=True,          # Prefix feature names with transformer name
        )

        # Log that the preprocessor has been built
        logger.info("Preprocessing pipeline built successfully")

        # Return the ColumnTransformer
        return preprocessor

    def _get_feature_names(self) -> List[str]:
        """
        Extract human-readable feature names after transformation.

        Returns
        -------
        list of str
            The names of all features after preprocessing.
        """
        # Get feature names from the fitted ColumnTransformer
        try:
            # sklearn >= 1.0 supports get_feature_names_out()
            feature_names = list(self.preprocessor.get_feature_names_out())
        except AttributeError:
            # Fallback: construct names manually for older sklearn versions
            # Get numerical feature names (unchanged)
            num_features = self.numerical_cols.copy()
            # Get one-hot encoded categorical feature names
            cat_encoder = self.preprocessor.named_transformers_["cat"].named_steps["encoder"]
            cat_features = list(cat_encoder.get_feature_names_out(self.categorical_cols))
            # Combine both lists
            feature_names = num_features + cat_features

        # Return the list of feature names
        return feature_names

    def get_feature_names(self) -> List[str]:
        """Public alias for _get_feature_names for external use."""
        return self._get_feature_names()

    def split_data(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets with stratification on the target.

        Stratification ensures both splits have the same churn rate,
        critical for imbalanced datasets.

        Parameters
        ----------
        df : pd.DataFrame
            The full preprocessed DataFrame with features and target.

        Returns
        -------
        tuple of (X_train, X_test, y_train, y_test)
            Feature DataFrames and target Series for train and test.
        """
        # Separate features (X) from the target variable (y)
        X = df.drop(columns=[self.target])         # All columns except target
        y = df[self.target]                         # Only the target column

        # Drop TotalCharges to avoid multicollinearity with tenure
        # (TotalCharges is highly correlated with tenure × MonthlyCharges)
        if 'TotalCharges' in X.columns:
            X = X.drop(columns=['TotalCharges'])
            # Also remove from numerical_cols list for the preprocessor
            if 'TotalCharges' in self.numerical_cols:
                self.numerical_cols = [c for c in self.numerical_cols if c != 'TotalCharges']
            logger.info("Dropped TotalCharges to avoid multicollinearity with tenure")

        # Perform stratified train/test split
        # stratify=y ensures both sets have the same churn rate proportion
        X_train, X_test, y_train, y_test = train_test_split(
            X,                                      # Feature matrix
            y,                                      # Target vector
            test_size=self.test_size,               # Fraction for test set
            random_state=self.random_state,         # Seed for reproducibility
            stratify=y,                             # Maintain class distribution
        )

        # Log the split sizes and class distributions
        logger.info(f"Data split: Train={len(X_train)}, Test={len(X_test)}")
        logger.info(f"  Train churn rate: {y_train.mean():.2%}")
        logger.info(f"  Test churn rate:  {y_test.mean():.2%}")

        # Return the four components of the split
        return X_train, X_test, y_train, y_test

    def fit_transform(
        self,
        X_train: pd.DataFrame,
        y: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, ...]:
        """
        Fit the preprocessor on training data and transform it.

        IMPORTANT: Only fit on training data to prevent data leakage.
        The test set must be transformed using .transform() only.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature DataFrame (no target column).

        Returns
        -------
        np.ndarray
            Transformed training features as a numpy array.
        """
        # If a target `y` is provided, assume caller expects a full split
        if y is not None:
            # Drop TotalCharges to avoid multicollinearity with tenure
            if 'TotalCharges' in X_train.columns:
                X_train = X_train.drop(columns=['TotalCharges'])
                # Also remove from numerical_cols list for the preprocessor
                if 'TotalCharges' in self.numerical_cols:
                    self.numerical_cols = [c for c in self.numerical_cols if c != 'TotalCharges']
                logger.info("Dropped TotalCharges to avoid multicollinearity with tenure")
            
            # Perform stratified train/test split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_train,
                y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y,
            )
            # Build and fit the preprocessor on the training portion
            self.preprocessor = self._build_preprocessor()
            X_tr_proc = self.preprocessor.fit_transform(X_tr)
            # Mark fitted and store names
            self.is_fitted = True
            self.feature_names = self._get_feature_names()
            # Transform the test set
            X_te_proc = self.preprocessor.transform(X_te)
            # Log and return four arrays (train/test split)
            logger.info("Preprocessor fitted and data split returned")
            return X_tr_proc, X_te_proc, y_tr, y_te

        # Drop TotalCharges to avoid multicollinearity with tenure
        if 'TotalCharges' in X_train.columns:
            X_train = X_train.drop(columns=['TotalCharges'])
            # Also remove from numerical_cols list for the preprocessor
            if 'TotalCharges' in self.numerical_cols:
                self.numerical_cols = [c for c in self.numerical_cols if c != 'TotalCharges']
            logger.info("Dropped TotalCharges to avoid multicollinearity with tenure")

        # Build the preprocessing pipeline for single-dataset fit
        self.preprocessor = self._build_preprocessor()
        # Fit the preprocessor on training data AND transform it
        X_train_processed = self.preprocessor.fit_transform(X_train)
        # Mark the preprocessor as fitted
        self.is_fitted = True
        # Extract and store feature names for later interpretability
        self.feature_names = self._get_feature_names()
        # Log the transformation results
        logger.info(f"Preprocessor fitted on training data")
        logger.info(f"  Input features: {X_train.shape[1]}")
        logger.info(f"  Output features: {X_train_processed.shape[1]}")
        logger.info(f"  Feature names: {self.feature_names[:5]}... (showing first 5)")
        # Return the transformed numpy array
        return X_train_processed

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using the already-fitted preprocessor.

        Use this for test data, validation data, or new predictions
        to ensure consistent transformations.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame to transform.

        Returns
        -------
        np.ndarray
            Transformed features as a numpy array.
        """
        # Check that the preprocessor has been fitted first
        if not self.is_fitted:
            # Raise an error if transform is called before fit
            raise RuntimeError(
                "Preprocessor has not been fitted yet. "
                "Call fit_transform(X_train) first."
            )

        # Transform the data using the fitted preprocessor
        X_processed = self.preprocessor.transform(X)

        # Log the transformation
        logger.info(f"Data transformed. Shape: {X_processed.shape}")

        # Return the transformed numpy array
        return X_processed

    def get_feature_dataframe(
        self,
        X_processed: np.ndarray,
    ) -> pd.DataFrame:
        """
        Convert the processed numpy array back into a labeled DataFrame.

        This is useful for SHAP explanations and interpretability,
        where feature names are needed.

        Parameters
        ----------
        X_processed : np.ndarray
            The transformed feature array from fit_transform/transform.

        Returns
        -------
        pd.DataFrame
            DataFrame with human-readable column names.
        """
        # Check that feature names are available
        if self.feature_names is None:
            # Raise an error if names aren't available
            raise RuntimeError("Feature names not available. Fit the preprocessor first.")

        # Create a DataFrame with the feature names as column headers
        df = pd.DataFrame(
            X_processed,                            # The numpy data
            columns=self.feature_names,             # Human-readable column names
        )

        # Return the labeled DataFrame
        return df

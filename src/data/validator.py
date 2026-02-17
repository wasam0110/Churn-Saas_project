# ============================================================
# src/data/validator.py
# Data quality validation: schema checks, missing values,
# cardinality checks, and data integrity verification.
# ============================================================

import pandas as pd                                # Pandas for DataFrame operations
import numpy as np                                 # NumPy for numerical checks
from loguru import logger                          # Structured logging
from typing import Dict, List, Tuple, Optional     # Type hints


class DataValidator:
    """
    Validates data quality and schema conformance.

    Runs a battery of checks on the raw/processed data to catch
    issues early before they corrupt downstream models.
    """

    def __init__(self, config: dict):
        """
        Initialize the validator with expected schema from config.

        Parameters
        ----------
        config : dict
            Project configuration containing expected column definitions.
        """
        # Store the full config for reference
        self.config = config
        # Extract expected numerical column names from config
        self.expected_numerical = config["features"]["numerical_columns"]
        # Extract expected categorical column names from config
        self.expected_categorical = config["features"]["categorical_columns"]
        # Extract the target column name
        self.target = config["features"]["target"]
        # Combine all expected columns into one list for schema checking
        self.expected_columns = (
            self.expected_numerical +               # Numerical features
            self.expected_categorical +             # Categorical features
            [self.target]                           # The target variable
        )
        # Initialize a list to collect validation issues
        self.issues: List[str] = []
        # Log initialization
        logger.info("DataValidator initialized with expected schema")

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Check that all expected columns are present in the DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.

        Returns
        -------
        bool
            True if all expected columns are present, False otherwise.
        """
        # Get the set of actual columns in the DataFrame
        actual_columns = set(df.columns)
        # Get the set of expected columns from config
        expected = set(self.expected_columns)
        # Find any columns that are expected but missing
        missing = expected - actual_columns
        # Find any unexpected extra columns in the data
        extra = actual_columns - expected

        # If there are missing columns, log and record the issue
        if missing:
            # Format the missing columns as a readable string
            issue = f"Missing columns: {sorted(missing)}"
            # Add to the issues list
            self.issues.append(issue)
            # Log the issue as a warning
            logger.warning(issue)

        # If there are extra columns, log them (informational, not an error)
        if extra:
            # These might be legitimate extra columns (e.g., customerID)
            logger.info(f"Extra columns found (not in config): {sorted(extra)}")

        # Return True only if no columns are missing
        return len(missing) == 0

    def validate_missing_values(self, df: pd.DataFrame, threshold: float = 0.3) -> Dict[str, float]:
        """
        Check for missing values in each column and flag those above threshold.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to check for missing values.
        threshold : float
            Maximum acceptable fraction of missing values (0.0 to 1.0).

        Returns
        -------
        dict
            Dictionary mapping column names to their missing value fractions.
        """
        # Calculate the fraction of missing values for each column
        missing_fractions = df.isnull().mean()
        # Filter to only columns that have any missing values
        missing_report = missing_fractions[missing_fractions > 0].to_dict()

        # Log each column's missing rate
        for col, frac in missing_report.items():
            # Check if the missing rate exceeds the acceptable threshold
            if frac > threshold:
                # Record a critical issue for high missing rates
                issue = f"Column '{col}' has {frac:.1%} missing values (exceeds {threshold:.0%} threshold)"
                # Add to issues list
                self.issues.append(issue)
                # Log as an error (this could break the model)
                logger.error(issue)
            else:
                # Log as informational for low missing rates
                logger.info(f"Column '{col}' has {frac:.1%} missing values (acceptable)")

        # Return the full missing values report
        return missing_report

    def validate_target_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze the target variable distribution for class imbalance.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the target column.

        Returns
        -------
        dict
            Dictionary with class counts and imbalance ratio.
        """
        # Check that the target column exists in the DataFrame
        if self.target not in df.columns:
            # Log error and return empty dict
            logger.error(f"Target column '{self.target}' not found in DataFrame")
            return {}

        # Count occurrences of each class (0 = retained, 1 = churned)
        value_counts = df[self.target].value_counts()
        # Calculate the total number of samples
        total = len(df)
        # Calculate churn rate (proportion of positive class)
        churn_rate = value_counts.get(1, 0) / total
        # Calculate the imbalance ratio (majority / minority)
        majority = value_counts.max()
        minority = value_counts.min()
        imbalance_ratio = majority / minority if minority > 0 else float("inf")

        # Build the distribution report
        report = {
            "total_samples": total,                # Total number of rows
            "churned": int(value_counts.get(1, 0)),  # Number of churned customers
            "retained": int(value_counts.get(0, 0)), # Number of retained customers
            "churn_rate": round(churn_rate, 4),    # Fraction that churned
            "imbalance_ratio": round(imbalance_ratio, 2),  # Majority/minority ratio
        }

        # Log the distribution summary
        logger.info(f"Target distribution: {report}")

        # Warn if the imbalance ratio is severe (>5:1)
        if imbalance_ratio > 5:
            # Record an issue about severe class imbalance
            issue = f"Severe class imbalance: {imbalance_ratio:.1f}:1 ratio"
            self.issues.append(issue)
            logger.warning(issue)

        # Return the distribution report
        return report

    def validate_numerical_ranges(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """
        Check numerical columns for unreasonable values (negatives, outliers).

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Returns
        -------
        dict
            Dictionary mapping column names to their statistics and flags.
        """
        # Initialize the report dictionary
        report = {}

        # Iterate over each expected numerical column
        for col in self.expected_numerical:
            # Skip if the column doesn't exist in the DataFrame
            if col not in df.columns:
                continue

            # Calculate basic statistics for this column
            stats = {
                "min": float(df[col].min()),           # Minimum value
                "max": float(df[col].max()),           # Maximum value
                "mean": float(df[col].mean()),         # Mean value
                "std": float(df[col].std()),           # Standard deviation
                "null_count": int(df[col].isnull().sum()),  # Number of nulls
            }

            # Flag negative values for columns that should be non-negative
            if stats["min"] < 0 and col in ["tenure", "MonthlyCharges", "TotalCharges"]:
                # These columns should never be negative
                issue = f"Column '{col}' has negative values (min={stats['min']})"
                self.issues.append(issue)
                logger.warning(issue)

            # Flag potential outliers using IQR method
            q1 = float(df[col].quantile(0.25))        # First quartile
            q3 = float(df[col].quantile(0.75))        # Third quartile
            iqr = q3 - q1                              # Interquartile range
            lower_bound = q1 - 3.0 * iqr              # Lower outlier boundary
            upper_bound = q3 + 3.0 * iqr              # Upper outlier boundary
            # Count values outside the IQR bounds
            outlier_count = int(
                ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            )
            # Add outlier info to stats
            stats["outlier_count"] = outlier_count

            # Log if significant outliers found
            if outlier_count > 0:
                logger.info(f"Column '{col}': {outlier_count} potential outliers detected")

            # Store this column's report
            report[col] = stats

        # Return the full numerical validation report
        return report

    def validate_categorical_cardinality(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Check cardinality (number of unique values) of categorical columns.
        High cardinality can cause issues with one-hot encoding.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        Returns
        -------
        dict
            Dictionary mapping column names to their unique value counts.
        """
        # Initialize the cardinality report
        cardinality = {}

        # Iterate over each expected categorical column
        for col in self.expected_categorical:
            # Skip if column doesn't exist
            if col not in df.columns:
                continue

            # Count unique values in this column
            n_unique = df[col].nunique()
            # Store in the report
            cardinality[col] = n_unique

            # Warn if cardinality is unexpectedly high (>20 unique values)
            if n_unique > 20:
                issue = f"Column '{col}' has high cardinality: {n_unique} unique values"
                self.issues.append(issue)
                logger.warning(issue)

            # Log the actual unique values for small cardinality
            if n_unique <= 10:
                unique_vals = df[col].unique().tolist()
                logger.debug(f"Column '{col}' values: {unique_vals}")

        # Return the cardinality report
        return cardinality

    def run_all_validations(self, df: pd.DataFrame) -> Dict:
        """
        Execute all validation checks and return a comprehensive report.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to validate.

        Returns
        -------
        dict
            Comprehensive validation report with all check results.
        """
        # Reset the issues list for a fresh validation run
        self.issues = []

        # Log the start of validation
        logger.info("=" * 60)
        logger.info("Starting comprehensive data validation...")
        logger.info("=" * 60)

        # Run all validation checks and collect results
        report = {
            "schema_valid": self.validate_schema(df),                   # Schema check
            "missing_values": self.validate_missing_values(df),         # Missing value check
            "target_distribution": self.validate_target_distribution(df), # Target balance check
            "numerical_ranges": self.validate_numerical_ranges(df),     # Range/outlier check
            "categorical_cardinality": self.validate_categorical_cardinality(df),  # Cardinality check
            "total_issues": len(self.issues),                           # Total issue count
            "issues": self.issues.copy(),                               # List of all issues found
        }

        # Log the validation summary
        if len(self.issues) == 0:
            # All checks passed
            logger.info("All validation checks PASSED. No issues found.")
        else:
            # Some issues were found
            logger.warning(f"Validation complete. {len(self.issues)} issue(s) found:")
            # Log each individual issue
            for i, issue in enumerate(self.issues, 1):
                logger.warning(f"  Issue {i}: {issue}")

        # Return the full validation report
        return report

    # --- Backwards-compatible aliases for older tests ---
    def check_schema(self, df: pd.DataFrame) -> bool:
        return self.validate_schema(df)

    def check_missing_values(self, df: pd.DataFrame, threshold: float = 0.3) -> bool:
        """
        Backwards-compatible wrapper that returns a boolean
        indicating whether missing-value checks passed.
        """
        report = self.validate_missing_values(df, threshold=threshold)
        # Pass if no columns exceed the threshold
        fails = [c for c, frac in report.items() if frac > threshold]
        return len(fails) == 0

    def check_target_distribution(self, df: pd.DataFrame) -> bool:
        """
        Backwards-compatible wrapper that returns a boolean
        indicating whether the target distribution check passed
        (no severe imbalance).
        """
        report = self.validate_target_distribution(df)
        # If report is empty (e.g., missing target), indicate failure
        if not report:
            return False
        # Consider imbalance_ratio > 5 as failing the check
        return float(report.get("imbalance_ratio", float("inf"))) <= 5.0

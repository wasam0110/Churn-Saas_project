# ============================================================
# src/features/engineer.py
# Advanced feature engineering pipeline for telco churn.
# Creates time-aware, interaction, and behavioral features
# that go beyond basic raw columns.
# ============================================================

import pandas as pd                                # Pandas for tabular data manipulation
import numpy as np                                 # NumPy for numerical computations
from loguru import logger                          # Structured logging
from typing import List, Optional                  # Type hints


class FeatureEngineer:
    """
    Creates advanced features from raw telco data to improve
    model performance and interpretability.

    Feature categories:
    1. Tenure-based segments and lifecycle features
    2. Charge ratios and spending patterns
    3. Service adoption and engagement scores
    4. Contract and billing risk indicators
    5. Interaction features between key variables
    """

    def __init__(self, config: dict):
        """
        Initialize the FeatureEngineer with project configuration.

        Parameters
        ----------
        config : dict
            Project configuration dictionary from config.yaml.
        """
        # Store the full configuration
        self.config = config
        # Extract rolling window sizes for time-aware features
        self.rolling_windows = config["features"].get("rolling_windows", [3, 6, 12])
        # Extract the target column name
        self.target = config["features"]["target"]
        # Track which new features are created (for documentation)
        self.created_features: List[str] = []
        # Log initialization
        logger.info("FeatureEngineer initialized")

    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tenure-based lifecycle features.

        Tenure is one of the strongest churn predictors. These features
        capture different aspects of the customer lifecycle.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with a 'tenure' column.

        Returns
        -------
        pd.DataFrame
            DataFrame with new tenure-based features added.
        """
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()

        # --- Tenure in Years ---
        # Convert months to years for more intuitive interpretation
        df["tenure_years"] = df["tenure"] / 12.0
        # Track this new feature
        self.created_features.append("tenure_years")

        # --- Tenure Segments ---
        # Bin customers into lifecycle stages using cut()
        # These segments have different churn risk profiles
        df["tenure_segment"] = pd.cut(
            df["tenure"],                              # The column to bin
            bins=[0, 6, 12, 24, 48, 72, float("inf")],  # Bin edges (months)
            labels=[                                    # Human-readable labels
                "0-6mo",                               # New customers (highest churn risk)
                "6-12mo",                              # Early relationship
                "1-2yr",                               # Building loyalty
                "2-4yr",                               # Established
                "4-6yr",                               # Loyal
                "6yr+",                                # Long-term (lowest churn risk)
            ],
            right=True,                                # Include right edge in bin
        )
        self.created_features.append("tenure_segment")

        # --- Backwards-compatible alias: tenure_group ---
        # Some older callers/tests expect a column named 'tenure_group'
        df["tenure_group"] = df["tenure_segment"]
        self.created_features.append("tenure_group")

        # --- Is New Customer Flag ---
        # Customers in their first 6 months have the highest churn risk
        df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
        self.created_features.append("is_new_customer")

        # --- Tenure Squared ---
        # Quadratic term captures non-linear relationship with churn
        # (churn risk decreases rapidly at first, then flattens)
        df["tenure_squared"] = df["tenure"] ** 2
        self.created_features.append("tenure_squared")

        # --- Log Tenure ---
        # Log transform compresses the long tail of high-tenure customers
        # Add 1 to avoid log(0) which is undefined
        df["tenure_log"] = np.log1p(df["tenure"])
        self.created_features.append("tenure_log")

        # Log the creation of tenure features
        logger.info("Created 5 tenure-based features")

        # Return the enriched DataFrame
        return df

    def create_charge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create spending pattern and charge ratio features.

        These features capture how much a customer pays relative
        to what they could be paying, and spending efficiency.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with charge-related columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with new charge-based features added.
        """
        # Create a copy to avoid modifying the original
        df = df.copy()

        # --- Average Monthly Charge ---
        # TotalCharges / tenure gives the actual average monthly spend
        # Handles division by zero for tenure=0 customers
        df["avg_monthly_charge"] = np.where(
            df["tenure"] > 0,                          # Condition: tenure > 0
            df["TotalCharges"] / df["tenure"],         # True: calculate ratio
            df["MonthlyCharges"],                      # False: use current monthly charge
        )
        self.created_features.append("avg_monthly_charge")

        # --- Charge Increase Indicator ---
        # If current monthly charge > historical average, price may have increased
        # Price increases are a known churn driver
        df["charge_increase"] = (
            df["MonthlyCharges"] > df["avg_monthly_charge"]
        ).astype(int)
        self.created_features.append("charge_increase")

        # --- Charge Difference ---
        # How much more (or less) the customer pays now vs their average
        df["charge_difference"] = df["MonthlyCharges"] - df["avg_monthly_charge"]
        self.created_features.append("charge_difference")

        # --- Charge per Service ---
        # Count total services the customer uses
        service_columns = [
            "PhoneService", "MultipleLines", "InternetService",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies",
        ]
        # Count 'Yes' values across service columns (some may have 'No internet service')
        existing_service_cols = [c for c in service_columns if c in df.columns]
        # Count how many services have a "Yes" value
        df["total_services"] = df[existing_service_cols].apply(
            lambda row: sum(1 for v in row if v == "Yes"),  # Count 'Yes' values
            axis=1,                                     # Apply across columns (row-wise)
        )
        self.created_features.append("total_services")

        # Monthly charge divided by number of services (value per service)
        # Avoid division by zero by using max(1, total_services)
        df["charge_per_service"] = df["MonthlyCharges"] / df["total_services"].clip(lower=1)
        self.created_features.append("charge_per_service")

        # --- Backwards-compatible alias: charge_per_tenure_month ---
        # Older tests expect this specific feature name; map to the
        # same value as 'avg_monthly_charge' which is safe (handles tenure=0).
        df["charge_per_tenure_month"] = df["avg_monthly_charge"]
        self.created_features.append("charge_per_tenure_month")

        # --- Lifetime Value Proxy (CLV) ---
        # Simple CLV estimate = MonthlyCharges * expected remaining tenure
        # Use tenure as a proxy for how long they might stay
        df["clv_proxy"] = df["MonthlyCharges"] * df["tenure"]
        self.created_features.append("clv_proxy")

        # --- Monthly Charges Tier ---
        # Segment customers by spending level
        df["charge_tier"] = pd.cut(
            df["MonthlyCharges"],                      # Column to bin
            bins=[0, 35, 70, 90, float("inf")],       # Price tier boundaries
            labels=["low", "medium", "high", "premium"],  # Tier labels
            right=True,                                # Include right edge
        )
        self.created_features.append("charge_tier")

        # Log feature creation
        logger.info("Created 7 charge-based features")

        # Return the enriched DataFrame
        return df

    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create service adoption and engagement features.

        Customers with more services and security features
        tend to be stickier (lower churn probability).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with service columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with new service-based features added.
        """
        # Create a copy to avoid modifying the original
        df = df.copy()

        # --- Has Internet Service Flag ---
        # Customers without internet have a very different profile
        df["has_internet"] = (df["InternetService"] != "No").astype(int)
        self.created_features.append("has_internet")

        # --- Internet Type: Fiber Optic Flag ---
        # Fiber optic customers churn more (counterintuitively—often due to price)
        df["is_fiber_optic"] = (df["InternetService"] == "Fiber optic").astype(int)
        self.created_features.append("is_fiber_optic")

        # --- Security/Protection Score ---
        # Count of security-related services the customer has
        # Customers with more protection services are stickier
        security_services = ["OnlineSecurity", "OnlineBackup",
                             "DeviceProtection", "TechSupport"]
        # Count how many security services are set to 'Yes'
        existing_security = [c for c in security_services if c in df.columns]
        df["security_score"] = df[existing_security].apply(
            lambda row: sum(1 for v in row if v == "Yes"),  # Count 'Yes' values
            axis=1,                                          # Apply row-wise
        )
        self.created_features.append("security_score")

        # --- Entertainment Score ---
        # Count of entertainment/streaming services
        entertainment_services = ["StreamingTV", "StreamingMovies"]
        existing_entertainment = [c for c in entertainment_services if c in df.columns]
        df["entertainment_score"] = df[existing_entertainment].apply(
            lambda row: sum(1 for v in row if v == "Yes"),
            axis=1,
        )
        self.created_features.append("entertainment_score")

        # --- Has No Protection Flag ---
        # Customers with zero security services are vulnerable to churn
        df["no_protection"] = (df["security_score"] == 0).astype(int)
        self.created_features.append("no_protection")

        # --- Service Diversity Ratio ---
        # Ratio of services used vs total possible services
        # Higher diversity = more engaged = lower churn risk
        total_possible = len(existing_security) + len(existing_entertainment)
        if total_possible > 0:
            # Calculate diversity as fraction of available services used
            df["service_diversity"] = (
                (df["security_score"] + df["entertainment_score"]) / total_possible
            )
        else:
            # If no service columns exist, set diversity to 0
            df["service_diversity"] = 0
        self.created_features.append("service_diversity")

        # Log feature creation
        logger.info("Created 6 service-based features")

        # Return the enriched DataFrame
        return df

    def create_contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create contract and billing risk indicator features.

        Month-to-month contracts and electronic check payments
        are strong churn predictors.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame with contract and billing columns.

        Returns
        -------
        pd.DataFrame
            DataFrame with new contract-based features added.
        """
        # Create a copy to avoid modifying the original
        df = df.copy()

        # --- Is Month-to-Month Contract ---
        # Month-to-month customers have the highest churn (no lock-in)
        df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
        self.created_features.append("is_month_to_month")

        # --- Has Long Contract ---
        # Two-year contracts have the lowest churn (strongest commitment)
        df["has_long_contract"] = (df["Contract"] == "Two year").astype(int)
        self.created_features.append("has_long_contract")

        # --- Uses Electronic Check ---
        # Electronic check users churn more (often less engaged/automated)
        df["uses_electronic_check"] = (
            df["PaymentMethod"] == "Electronic check"
        ).astype(int)
        self.created_features.append("uses_electronic_check")

        # --- Uses Auto-Pay ---
        # Automatic payment methods indicate higher commitment
        auto_pay_methods = ["Bank transfer (automatic)", "Credit card (automatic)"]
        df["uses_auto_pay"] = df["PaymentMethod"].isin(auto_pay_methods).astype(int)
        self.created_features.append("uses_auto_pay")

        # --- Paperless + Electronic Check Risk ---
        # Combination of paperless billing AND electronic check is high-risk
        df["paperless_echeck_risk"] = (
            (df["PaperlessBilling"] == "Yes") &        # Has paperless billing AND
            (df["PaymentMethod"] == "Electronic check") # Uses electronic check
        ).astype(int)
        self.created_features.append("paperless_echeck_risk")

        # Log feature creation
        logger.info("Created 5 contract-based features")

        # Return the enriched DataFrame
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features that combine multiple columns.

        Interaction features capture synergies: e.g., a new customer
        on a month-to-month contract with no protection is very high risk.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame (should have engineered features from above).

        Returns
        -------
        pd.DataFrame
            DataFrame with new interaction features added.
        """
        # Create a copy to avoid modifying the original
        df = df.copy()

        # --- High Risk Combo: New + Month-to-Month + No Protection ---
        # Triple threat: new customer, no contract lock-in, no security services
        if all(c in df.columns for c in ["is_new_customer", "is_month_to_month", "no_protection"]):
            df["high_risk_combo"] = (
                df["is_new_customer"] *                # Is new (0 or 1)
                df["is_month_to_month"] *              # No contract (0 or 1)
                df["no_protection"]                    # No security (0 or 1)
            )
            self.created_features.append("high_risk_combo")

        # --- Tenure x Monthly Charges Interaction ---
        # Captures the idea that high charges hurt more for new customers
        df["tenure_x_monthly"] = df["tenure"] * df["MonthlyCharges"]
        self.created_features.append("tenure_x_monthly")

        # --- Contract x Charges Interaction ---
        # Month-to-month + high charges = highest churn risk
        if "is_month_to_month" in df.columns:
            df["mtm_x_charges"] = df["is_month_to_month"] * df["MonthlyCharges"]
            self.created_features.append("mtm_x_charges")

        # --- Fiber + No Security Interaction ---
        # Fiber optic customers without security churn very frequently
        if all(c in df.columns for c in ["is_fiber_optic", "no_protection"]):
            df["fiber_no_security"] = df["is_fiber_optic"] * df["no_protection"]
            self.created_features.append("fiber_no_security")

        # --- Senior x Month-to-Month ---
        # Senior citizens on month-to-month contracts are particularly vulnerable
        if "is_month_to_month" in df.columns and "SeniorCitizen" in df.columns:
            df["senior_mtm"] = df["SeniorCitizen"] * df["is_month_to_month"]
            self.created_features.append("senior_mtm")

        # --- Service Engagement x Tenure ---
        # Long-tenure customers with high engagement are the safest
        if "service_diversity" in df.columns:
            df["engagement_x_tenure"] = df["service_diversity"] * df["tenure"]
            self.created_features.append("engagement_x_tenure")

        # Log feature creation
        logger.info("Created interaction features")

        # Return the enriched DataFrame
        return df

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.

        This is the main entry point. It applies all feature creation
        methods in the correct order.

        Parameters
        ----------
        df : pd.DataFrame
            Raw or lightly processed DataFrame.

        Returns
        -------
        pd.DataFrame
            Fully enriched DataFrame with all engineered features.
        """
        # Reset the list of created features for this run
        self.created_features = []

        # Log the start of feature engineering
        logger.info("=" * 60)
        logger.info("Starting feature engineering pipeline...")
        logger.info(f"  Input shape: {df.shape}")

        # Step 1: Create tenure-based lifecycle features
        df = self.create_tenure_features(df)

        # Step 2: Create charge and spending pattern features
        df = self.create_charge_features(df)

        # Step 3: Create service adoption features
        df = self.create_service_features(df)

        # Step 4: Create contract and billing risk features
        df = self.create_contract_features(df)

        # Step 5: Create interaction features (depends on above)
        df = self.create_interaction_features(df)

        # Log the summary of feature engineering
        logger.info(f"  Output shape: {df.shape}")
        logger.info(f"  New features created: {len(self.created_features)}")
        logger.info(f"  Feature list: {self.created_features}")
        logger.info("Feature engineering pipeline complete")
        logger.info("=" * 60)

        # Return the fully enriched DataFrame
        return df

    # Backwards-compatible alias for older tests/code
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alias for `engineer_all_features` to maintain compatibility
        with older callers/tests that expect `create_all_features`.
        """
        return self.engineer_all_features(df)

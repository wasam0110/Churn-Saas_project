# ============================================================
# scripts/train.py
# End-to-end training pipeline orchestrator.
# Loads data, engineers features, trains models, evaluates,
# generates explanations, and registers the best model.
# ============================================================

import sys                                         # System-specific parameters
from pathlib import Path                           # Object-oriented file paths

# Add the project root to Python path for module imports
project_root = Path(__file__).resolve().parent.parent  # Navigate up from scripts/
sys.path.insert(0, str(project_root))              # Insert at beginning of path

import warnings                                    # Warning control module
warnings.filterwarnings("ignore")                  # Suppress non-critical warnings

import pandas as pd                                # Data manipulation library
import numpy as np                                 # Numerical computing library
import joblib                                      # Serialization for Python objects
from loguru import logger                          # Structured logging library
from datetime import datetime                      # Date and time utilities

# Import project modules
from src.utils.helpers import (                    # Utility functions
    load_config,                                   # YAML config loader
    setup_logging,                                 # Logging configuration
    ensure_directory,                              # Directory creator
    save_model,                                    # Model serializer
)
from src.data.loader import DataLoader             # Data loading module
from src.data.validator import DataValidator        # Data validation module
from src.data.preprocessor import DataPreprocessor  # Data preprocessing module
from src.features.engineer import FeatureEngineer   # Feature engineering module
from src.features.selector import FeatureSelector   # Feature selection module
from src.models.trainer import ModelTrainer         # Model training module
from src.models.evaluator import ModelEvaluator     # Model evaluation module
from src.models.explainer import ModelExplainer     # SHAP explainability module
from src.models.registry import ModelRegistry       # Model registry module


def run_training_pipeline():
    """
    Execute the full training pipeline from data loading to model registration.

    Steps:
    1. Load and validate data
    2. Engineer features
    3. Preprocess (encode, scale, split)
    4. Select features
    5. Train multiple models with calibration
    6. Evaluate all models
    7. Generate SHAP explanations for the best model
    8. Register the best model in the model registry
    """
    # ----------------------------------------------------------
    # Step 0: Setup
    # ----------------------------------------------------------
    # Load the project configuration
    config = load_config()
    # Setup logging with the configured log directory
    setup_logging(config.get("logging", {}).get("dir", "logs"))
    # Log the start of the pipeline
    logger.info("=" * 60)
    logger.info("CHURN PREDICTION TRAINING PIPELINE")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Ensure all output directories exist
    ensure_directory("models")                     # For saved models
    ensure_directory("reports")                    # For evaluation reports
    ensure_directory("reports/monitoring")          # For drift reports
    ensure_directory("data/processed")             # For processed datasets

    # ----------------------------------------------------------
    # Step 1: Load Data
    # ----------------------------------------------------------
    logger.info("Step 1: Loading data...")
    # Create a DataLoader instance with the config
    loader = DataLoader(config)
    # Get the raw data file path from config
    data_path = config.get("data", {}).get("raw_path", "data/raw/telco_churn.csv")

    # Check if the data file exists
    if not Path(data_path).exists():
        # If no real data, create synthetic data for demonstration
        logger.warning(f"Data file not found at {data_path}. Generating synthetic data...")
        # Generate synthetic data
        df = generate_synthetic_data()
        # Create parent directory if needed
        ensure_directory(Path(data_path).parent)
        # Save the synthetic data
        df.to_csv(data_path, index=False)
        logger.info(f"Synthetic data saved to {data_path}")
    else:
        # Load the real data from CSV
        df = loader.load_csv(data_path)

    # Log the shape of the loaded data
    logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # ----------------------------------------------------------
    # Step 2: Validate Data
    # ----------------------------------------------------------
    logger.info("Step 2: Validating data...")
    # Create a DataValidator instance
    validator = DataValidator(config)
    # Run all validation checks
    validation_results = validator.run_all_validations(df)
    # Log each validation result
    for check, passed in validation_results.items():
        # Use different log levels for passed/failed checks
        if passed:
            logger.info(f"  ✅ {check}: PASSED")
        else:
            logger.warning(f"  ⚠️ {check}: FAILED")

    # ----------------------------------------------------------
    # Step 3: Feature Engineering
    # ----------------------------------------------------------
    logger.info("Step 3: Engineering features...")
    # Create a FeatureEngineer instance with the config
    engineer = FeatureEngineer(config)
    # Apply all feature engineering transformations
    df_featured = engineer.engineer_all_features(df)
    # Log the number of new features created
    new_features = df_featured.shape[1] - df.shape[1]
    logger.info(f"Created {new_features} new features. Total columns: {df_featured.shape[1]}")

    # ----------------------------------------------------------
    # Step 4: Preprocessing
    # ----------------------------------------------------------
    logger.info("Step 4: Preprocessing data...")
    # Create a DataPreprocessor instance
    preprocessor = DataPreprocessor(config)

    # Get the target column name from config (features.target)
    target_col = config.get("features", {}).get("target", "Churn")
    # Ensure target is binary (0/1)
    if df_featured[target_col].dtype == object:
        # Convert string labels to binary integers
        df_featured[target_col] = (df_featured[target_col] == "Yes").astype(int)

    # Ensure target column exists and is binary (0/1)
    id_col = config.get("data", {}).get("id_column", "customerID")
    if target_col not in df_featured.columns:
        raise KeyError(f"Target column '{target_col}' not found in data")
    if df_featured[target_col].dtype == object:
        df_featured[target_col] = (df_featured[target_col] == "Yes").astype(int)

    # Log class distribution
    churn_rate = df_featured[target_col].mean()
    logger.info(f"Target distribution: {churn_rate:.1%} churn, {1-churn_rate:.1%} no churn")

    # Split into train/test using the preprocessor helper (stratified)
    X_train_df, X_test_df, y_train, y_test = preprocessor.split_data(df_featured)
    logger.info(f"Train set: {len(X_train_df)} samples, Test set: {len(X_test_df)} samples")

    # Fit preprocessor on training features and transform both train and test
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)
    logger.info(f"Features after preprocessing: {X_train.shape[1]}")

    # Save the fitted preprocessor for later use
    joblib.dump(preprocessor, "models/preprocessor.joblib")
    logger.info("Preprocessor saved to models/preprocessor.joblib")

    # ----------------------------------------------------------
    # Step 5: Feature Selection (Optional)
    # ----------------------------------------------------------
    logger.info("Step 5: Selecting features...")
    # Create a FeatureSelector instance
    selector = FeatureSelector(config)
    # Get feature names from the preprocessor
    try:
        # Try to get feature names from the preprocessor
        feature_names = preprocessor._get_feature_names()
    except Exception:
        # Fall back to generic names if not available
        feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]

    # Create DataFrames with feature names for the selector
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    # Select the most important features (returns list)
    selected_features = selector.select_features(X_train_df, y_train)
    logger.info(f"Selected {len(selected_features)} features out of {len(feature_names)}")

    # For training we need the filtered DataFrame; use the helper to get both
    X_train_filtered_df, _ = selector.select_features_df(X_train_df, y_train)
    # Filter to selected features for both train and test
    X_train_selected = X_train_filtered_df.values
    X_test_selected = X_test_df[selected_features].values

    # Save selected feature names
    joblib.dump(selected_features, "models/selected_features.joblib")
    logger.info("Selected features saved to models/selected_features.joblib")

    # ----------------------------------------------------------
    # Step 6: Model Training
    # ----------------------------------------------------------
    logger.info("Step 6: Training models...")
    # Create a ModelTrainer instance
    trainer = ModelTrainer(config)
    # Train all configured models
    trained_models = trainer.train_all_models(X_train_selected, y_train)
    # Log the number of trained models
    logger.info(f"Trained {len(trained_models)} models")

    # Calibrate probabilities for each model
    logger.info("Calibrating model probabilities...")
    calibrated_models = {}                         # Dictionary for calibrated models
    for name, model in trained_models.items():
        try:
            # Calibrate the model using the configured method (trainer uses model name)
            cal_model = trainer.calibrate_model(name, X_train_selected, y_train)
            calibrated_models[name] = cal_model
            logger.info(f"  ✅ {name}: calibrated successfully")
        except Exception as e:
            # If calibration fails, keep the uncalibrated model
            logger.warning(f"  ⚠️ {name}: calibration failed ({e}), using uncalibrated")
            calibrated_models[name] = model

    # Find optimal classification threshold for each model
    logger.info("Optimizing classification thresholds...")
    thresholds = {}                                # Dictionary for optimal thresholds
    for name, model in calibrated_models.items():
        try:
            # Obtain predicted probabilities for the positive class
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_test_selected)[:, 1]
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_test_selected)
                proba = 1 / (1 + np.exp(-scores))
            else:
                raise ValueError("Model has neither predict_proba nor decision_function")

            # Optimize threshold based on expected profit (y_true, y_proba)
            opt_thr, opt_profit = trainer.find_optimal_threshold(y_test, proba)
            thresholds[name] = opt_thr
            logger.info(f"  {name}: optimal threshold = {opt_thr:.3f}")
        except Exception as e:
            logger.warning(f"  ⚠️ {name}: threshold optimization failed ({e}), using 0.5")
            thresholds[name] = 0.5

    # ----------------------------------------------------------
    # Step 7: Model Evaluation
    # ----------------------------------------------------------
    logger.info("Step 7: Evaluating models...")
    # Create a ModelEvaluator instance
    evaluator = ModelEvaluator(config)

    # Evaluate each model and collect results
    all_results = {}                               # Dictionary for all evaluation results
    for name, model in calibrated_models.items():
        # Compute comprehensive metrics
        metrics = evaluator.compute_metrics(model, X_test_selected, y_test)
        all_results[name] = metrics
        # Log key metrics
        logger.info(f"  {name}:")
        logger.info(f"    ROC-AUC: {metrics.get('roc_auc', 0):.4f}")
        logger.info(f"    PR-AUC:  {metrics.get('pr_auc', 0):.4f}")
        logger.info(f"    F1:      {metrics.get('f1', 0):.4f}")

    # Generate comparison plots
    try:
        evaluator.compare_models(calibrated_models, X_test_selected, y_test)
        logger.info("Evaluation plots saved to reports/")
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")

    # Determine the best model by ROC-AUC
    best_model_name = max(all_results, key=lambda k: all_results[k].get("roc_auc", 0))
    best_model = calibrated_models[best_model_name]
    best_threshold = thresholds[best_model_name]
    best_metrics = all_results[best_model_name]
    logger.info(f"Best model: {best_model_name} (ROC-AUC: {best_metrics['roc_auc']:.4f})")

    # ----------------------------------------------------------
    # Step 8: Explainability
    # ----------------------------------------------------------
    logger.info("Step 8: Generating SHAP explanations...")
    try:
        # Create a ModelExplainer instance
        explainer = ModelExplainer(config)
        # Compute SHAP values using a sample of test data
        sample_size = min(500, X_test_selected.shape[0])
        X_sample = X_test_selected[:sample_size]
        # Compute SHAP values
        shap_values = explainer.compute_shap_values(best_model, X_sample, selected_features)
        # Get global feature importance
        importance = explainer.get_global_importance()
        logger.info("Top 10 most important features:")
        for feat, imp in list(importance.items())[:10]:
            logger.info(f"  {feat}: {imp:.4f}")
        # Save summary plot
        explainer.plot_summary(save_path="reports/shap_summary.png")
        logger.info("SHAP summary plot saved to reports/shap_summary.png")
    except Exception as e:
        logger.warning(f"SHAP explanation generation failed: {e}")

    # ----------------------------------------------------------
    # Step 9: Save Best Model
    # ----------------------------------------------------------
    logger.info("Step 9: Saving the best model...")
    # Save the best model using the helper function
    save_model(best_model, "models/best_model.joblib")
    # Save the optimal threshold
    joblib.dump(best_threshold, "models/optimal_threshold.joblib")
    logger.info(f"Best model saved: models/best_model.joblib")
    logger.info(f"Optimal threshold saved: {best_threshold:.3f}")

    # ----------------------------------------------------------
    # Step 10: Register Model
    # ----------------------------------------------------------
    logger.info("Step 10: Registering model...")
    try:
        # Create a ModelRegistry instance
        registry = ModelRegistry(config)
        # Register the best model with metadata
        version = registry.register_model(
            model=best_model,
            model_name=best_model_name,
            metrics=best_metrics,
            config=config,
        )
        # Promote the newly registered model to production
        registry.promote_to_production(best_model_name, version)
        logger.info(f"Model registered as {best_model_name} v{version} (production)")
    except Exception as e:
        logger.warning(f"Model registration failed: {e}")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info(f"Best Model: {best_model_name}")
    logger.info(f"ROC-AUC: {best_metrics.get('roc_auc', 0):.4f}")
    logger.info(f"PR-AUC: {best_metrics.get('pr_auc', 0):.4f}")
    logger.info(f"Threshold: {best_threshold:.3f}")
    logger.info(f"Completed at: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # Return the best model and its metadata
    return {
        "model_name": best_model_name,
        "model": best_model,
        "threshold": best_threshold,
        "metrics": best_metrics,
    }


def generate_synthetic_data(n_samples=7043, random_state=42):
    """
    Generate synthetic Telco Customer Churn data for demonstration.

    Parameters
    ----------
    n_samples : int, default=7043
        Number of customers to generate.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame mimicking the Telco Customer Churn dataset.
    """
    # Set the random seed for reproducibility
    np.random.seed(random_state)
    # Log that we are generating synthetic data
    logger.info(f"Generating {n_samples} synthetic customer records...")

    # Generate customer IDs
    customer_ids = [f"CUST-{i:05d}" for i in range(n_samples)]

    # Generate demographic features
    gender = np.random.choice(["Male", "Female"], n_samples)
    senior = np.random.choice([0, 1], n_samples, p=[0.84, 0.16])
    partner = np.random.choice(["Yes", "No"], n_samples, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], n_samples, p=[0.30, 0.70])

    # Generate tenure (months) with a bimodal distribution
    tenure = np.concatenate([
        np.random.randint(1, 12, n_samples // 3),        # Short tenure
        np.random.randint(12, 48, n_samples // 3),       # Medium tenure
        np.random.randint(48, 73, n_samples - 2 * (n_samples // 3)),  # Long tenure
    ])
    np.random.shuffle(tenure)                      # Shuffle to mix them

    # Generate service features
    phone_service = np.random.choice(["Yes", "No"], n_samples, p=[0.90, 0.10])
    multiple_lines = np.where(
        phone_service == "No",
        "No phone service",
        np.random.choice(["Yes", "No"], n_samples),
    )
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], n_samples, p=[0.34, 0.44, 0.22]
    )

    # Internet-dependent services
    def internet_dep(n, internet):
        """Generate internet-dependent service values."""
        return np.where(
            internet == "No",
            "No internet service",
            np.random.choice(["Yes", "No"], n),
        )

    online_security = internet_dep(n_samples, internet_service)
    online_backup = internet_dep(n_samples, internet_service)
    device_protection = internet_dep(n_samples, internet_service)
    tech_support = internet_dep(n_samples, internet_service)
    streaming_tv = internet_dep(n_samples, internet_service)
    streaming_movies = internet_dep(n_samples, internet_service)

    # Contract type
    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        n_samples,
        p=[0.55, 0.21, 0.24],
    )

    # Billing features
    paperless = np.random.choice(["Yes", "No"], n_samples, p=[0.60, 0.40])
    payment = np.random.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        n_samples,
        p=[0.34, 0.23, 0.22, 0.21],
    )

    # Monthly charges (correlated with services)
    base_charge = 20.0                             # Base monthly charge
    monthly_charges = base_charge + np.random.normal(0, 5, n_samples)
    # Add charge for fiber optic
    monthly_charges += np.where(internet_service == "Fiber optic", 30, 0)
    # Add charge for DSL
    monthly_charges += np.where(internet_service == "DSL", 15, 0)
    # Add charges for additional services
    for svc in [online_security, online_backup, device_protection,
                tech_support, streaming_tv, streaming_movies]:
        monthly_charges += np.where(svc == "Yes", np.random.uniform(5, 15, n_samples), 0)
    # Clip to realistic range
    monthly_charges = np.clip(monthly_charges, 18.25, 118.75)
    # Round to 2 decimals
    monthly_charges = np.round(monthly_charges, 2)

    # Total charges (tenure * monthly + noise)
    total_charges = tenure * monthly_charges + np.random.normal(0, 50, n_samples)
    total_charges = np.maximum(total_charges, monthly_charges)
    total_charges = np.round(total_charges, 2)

    # Generate churn label (correlated with features)
    churn_prob = np.zeros(n_samples)               # Base churn probability
    # Higher churn for month-to-month contracts
    churn_prob += np.where(contract == "Month-to-month", 0.25, 0)
    # Lower churn for two-year contracts
    churn_prob += np.where(contract == "Two year", -0.15, 0)
    # Higher churn for short tenure
    churn_prob += np.where(tenure < 12, 0.15, 0)
    # Lower churn for long tenure
    churn_prob += np.where(tenure > 48, -0.10, 0)
    # Higher churn for fiber optic (expensive)
    churn_prob += np.where(internet_service == "Fiber optic", 0.10, 0)
    # Higher churn for electronic check payment
    churn_prob += np.where(payment == "Electronic check", 0.10, 0)
    # Higher churn for high monthly charges
    churn_prob += np.where(monthly_charges > 70, 0.08, 0)
    # Lower churn for tech support
    churn_prob += np.where(tech_support == "Yes", -0.05, 0)
    # Lower churn for online security
    churn_prob += np.where(online_security == "Yes", -0.05, 0)
    # Add noise
    churn_prob += np.random.normal(0, 0.05, n_samples)
    # Clip to valid probability range
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    # Generate binary churn label
    churn = np.where(np.random.random(n_samples) < churn_prob, "Yes", "No")

    # Assemble the DataFrame
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn,
    })

    # Log the churn distribution
    churn_rate = (churn == "Yes").mean()
    logger.info(f"Synthetic data churn rate: {churn_rate:.1%}")

    # Return the completed DataFrame
    return df


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    # Run the full training pipeline
    result = run_training_pipeline()
    # Print the final summary
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best model: {result['model_name']}")
    print(f"ROC-AUC: {result['metrics'].get('roc_auc', 0):.4f}")
    print(f"Threshold: {result['threshold']:.3f}")
    print(f"{'='*60}")

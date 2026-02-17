# ============================================================
# scripts/predict.py
# Batch prediction script for the Churn Prediction SaaS.
# Loads a trained model, preprocesses new customer data,
# generates predictions, and outputs results to CSV.
# ============================================================

import sys                                         # System-specific parameters
from pathlib import Path                           # Object-oriented file paths

# Add the project root to Python path for module imports
project_root = Path(__file__).resolve().parent.parent  # Navigate up from scripts/
sys.path.insert(0, str(project_root))              # Insert at beginning of path

import argparse                                    # Command-line argument parsing
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
    load_model,                                    # Model loader
    ensure_directory,                              # Directory creator
)
from src.features.engineer import FeatureEngineer   # Feature engineering module
from src.actions.recommender import ActionRecommender  # Action recommendation module


def parse_args():
    """
    Parse command-line arguments for the prediction script.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with input_file, output_file, and threshold.
    """
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="Batch churn prediction script"  # Help text
    )

    # Input file path argument (required)
    parser.add_argument(
        "--input", "-i",                           # Short and long flag
        type=str,                                  # String type
        required=True,                             # Must be provided
        help="Path to the input CSV file with customer data",
    )

    # Output file path argument (optional, has default)
    parser.add_argument(
        "--output", "-o",                          # Short and long flag
        type=str,                                  # String type
        default="data/processed/predictions.csv",  # Default output path
        help="Path to save the predictions CSV (default: data/processed/predictions.csv)",
    )

    # Model path argument (optional)
    parser.add_argument(
        "--model", "-m",                           # Short and long flag
        type=str,                                  # String type
        default="models/best_model.joblib",        # Default model path
        help="Path to the trained model file (default: models/best_model.joblib)",
    )

    # Classification threshold argument (optional)
    parser.add_argument(
        "--threshold", "-t",                       # Short and long flag
        type=float,                                # Float type
        default=None,                              # Will load from saved threshold
        help="Classification threshold (default: loaded from saved threshold)",
    )

    # Include retention actions flag (optional)
    parser.add_argument(
        "--actions",                               # Long flag only
        action="store_true",                       # Boolean flag
        help="Include retention action recommendations in output",
    )

    # Parse and return the arguments
    return parser.parse_args()


def load_prediction_resources(model_path):
    """
    Load all resources needed for prediction.

    Parameters
    ----------
    model_path : str
        Path to the trained model file.

    Returns
    -------
    tuple
        (model, preprocessor, selected_features, threshold)
    """
    # Load the trained model
    logger.info(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Load the preprocessor
    preprocessor_path = "models/preprocessor.joblib"
    if Path(preprocessor_path).exists():
        logger.info(f"Loading preprocessor from {preprocessor_path}...")
        preprocessor = joblib.load(preprocessor_path)
    else:
        # Warn if preprocessor not found
        logger.warning("Preprocessor not found. Predictions may fail.")
        preprocessor = None

    # Load selected feature names
    features_path = "models/selected_features.joblib"
    if Path(features_path).exists():
        logger.info(f"Loading selected features from {features_path}...")
        selected_features = joblib.load(features_path)
    else:
        # No feature selection was used
        logger.info("No saved feature selection found. Using all features.")
        selected_features = None

    # Load the optimal threshold
    threshold_path = "models/optimal_threshold.joblib"
    if Path(threshold_path).exists():
        threshold = joblib.load(threshold_path)
        logger.info(f"Loaded optimal threshold: {threshold:.3f}")
    else:
        # Fall back to default 0.5
        threshold = 0.5
        logger.info(f"Using default threshold: {threshold:.3f}")

    # Return all loaded resources
    return model, preprocessor, selected_features, threshold


def assign_risk_level(probability, threshold):
    """
    Assign a human-readable risk level based on churn probability.

    Parameters
    ----------
    probability : float
        The predicted churn probability (0 to 1).
    threshold : float
        The classification threshold.

    Returns
    -------
    str
        One of 'LOW', 'MEDIUM', or 'HIGH'.
    """
    # High risk: above the optimal threshold
    if probability >= threshold:
        return "HIGH"
    # Medium risk: between half the threshold and the threshold
    elif probability >= threshold * 0.6:
        return "MEDIUM"
    # Low risk: below half the threshold
    else:
        return "LOW"


def run_batch_prediction(args):
    """
    Execute the batch prediction pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    # ----------------------------------------------------------
    # Setup
    # ----------------------------------------------------------
    # Load the project configuration
    config = load_config()
    # Setup logging
    setup_logging(config.get("logging", {}).get("dir", "logs"))
    # Log the start
    logger.info("=" * 60)
    logger.info("BATCH PREDICTION PIPELINE")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output file: {args.output}")
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # Step 1: Load Resources
    # ----------------------------------------------------------
    # Load the model, preprocessor, features, and threshold
    model, preprocessor, selected_features, threshold = load_prediction_resources(args.model)

    # Override threshold if specified via command line
    if args.threshold is not None:
        threshold = args.threshold
        logger.info(f"Using command-line threshold: {threshold:.3f}")

    # ----------------------------------------------------------
    # Step 2: Load Input Data
    # ----------------------------------------------------------
    logger.info("Loading input data...")
    # Check that the input file exists
    if not Path(args.input).exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)                                # Exit with error code

    # Read the CSV file
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} customers from {args.input}")

    # Save customer IDs for the output
    id_col = config.get("data", {}).get("id_column", "customerID")
    if id_col in df.columns:
        customer_ids = df[id_col].copy()           # Save IDs
    else:
        # Generate sequential IDs if not present
        customer_ids = pd.Series([f"CUST-{i}" for i in range(len(df))])

    # ----------------------------------------------------------
    # Step 3: Feature Engineering
    # ----------------------------------------------------------
    logger.info("Engineering features...")
    # Create feature engineer and apply transformations
    engineer = FeatureEngineer(config)
    df_featured = engineer.create_all_features(df)
    logger.info(f"Features after engineering: {df_featured.shape[1]}")

    # ----------------------------------------------------------
    # Step 4: Preprocessing
    # ----------------------------------------------------------
    logger.info("Preprocessing data...")
    # Remove non-feature columns
    target_col = config.get("data", {}).get("target_column", "Churn")
    drop_cols = [col for col in [id_col, target_col] if col in df_featured.columns]
    X = df_featured.drop(columns=drop_cols, errors="ignore")

    # Transform using the fitted preprocessor
    if preprocessor is not None:
        X_processed = preprocessor.transform(X)
    else:
        # If no preprocessor, use raw features
        X_processed = X.values

    # Apply feature selection if applicable
    if selected_features is not None:
        try:
            # Get feature names from preprocessor, with a robust fallback.
            try:
                feature_names = preprocessor._get_feature_names()
            except Exception:
                feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

            if not any(f in feature_names for f in selected_features):
                feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]

            X_df = pd.DataFrame(X_processed, columns=feature_names)
            available = [f for f in selected_features if f in X_df.columns]
            if available:
                X_processed = X_df[available].values
        except Exception as e:
            logger.warning(f"Feature selection failed: {e}. Using all features.")

    logger.info(f"Final feature matrix shape: {X_processed.shape}")

    # ----------------------------------------------------------
    # Step 5: Generate Predictions
    # ----------------------------------------------------------
    logger.info("Generating predictions...")
    # Get churn probabilities from the model
    probabilities = model.predict_proba(X_processed)[:, 1]
    # Apply the threshold to get binary predictions
    predictions = (probabilities >= threshold).astype(int)
    # Assign risk levels
    risk_levels = [assign_risk_level(p, threshold) for p in probabilities]

    # Log prediction summary
    n_high = sum(1 for r in risk_levels if r == "HIGH")
    n_medium = sum(1 for r in risk_levels if r == "MEDIUM")
    n_low = sum(1 for r in risk_levels if r == "LOW")
    logger.info(f"Predictions: HIGH={n_high}, MEDIUM={n_medium}, LOW={n_low}")

    # ----------------------------------------------------------
    # Step 6: Build Output DataFrame
    # ----------------------------------------------------------
    logger.info("Building output DataFrame...")
    # Create the output DataFrame with predictions
    output_df = pd.DataFrame({
        "customer_id": customer_ids,               # Customer identifier
        "churn_probability": np.round(probabilities, 4),  # Predicted probability
        "churn_prediction": predictions,           # Binary prediction (0/1)
        "risk_level": risk_levels,                 # Human-readable risk level
        "threshold_used": threshold,               # The threshold applied
        "prediction_date": datetime.now().isoformat(),  # Timestamp of prediction
    })

    # ----------------------------------------------------------
    # Step 7: Add Retention Actions (if requested)
    # ----------------------------------------------------------
    if args.actions:
        logger.info("Generating retention action recommendations...")
        try:
            # Create an ActionRecommender instance
            recommender = ActionRecommender(config)
            # Generate recommendations for high-risk customers
            actions = []
            for i, row in output_df.iterrows():
                if row["risk_level"] in ["HIGH", "MEDIUM"]:
                    # Get customer features for recommendations
                    customer_data = df.iloc[i].to_dict() if i < len(df) else {}
                    # Get the top recommended action
                    recs = recommender.recommend_actions(
                        churn_probability=row["churn_probability"],
                        customer_features=customer_data,
                        top_k=1,                   # Only the top action
                    )
                    if recs:
                        actions.append(recs[0].get("action", "Review account"))
                    else:
                        actions.append("Review account")
                else:
                    # Low risk: no immediate action needed
                    actions.append("No action needed")
            # Add the actions column
            output_df["recommended_action"] = actions
            logger.info("Retention actions added to output.")
        except Exception as e:
            logger.warning(f"Action recommendation failed: {e}")

    # ----------------------------------------------------------
    # Step 8: Save Results
    # ----------------------------------------------------------
    # Ensure the output directory exists
    ensure_directory(Path(args.output).parent)
    # Save to CSV
    output_df.to_csv(args.output, index=False)
    logger.info(f"Predictions saved to {args.output}")

    # ----------------------------------------------------------
    # Print Summary
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("BATCH PREDICTION COMPLETE")
    logger.info(f"Total customers: {len(output_df)}")
    logger.info(f"High risk: {n_high} ({n_high/len(output_df)*100:.1f}%)")
    logger.info(f"Medium risk: {n_medium} ({n_medium/len(output_df)*100:.1f}%)")
    logger.info(f"Low risk: {n_low} ({n_low/len(output_df)*100:.1f}%)")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Completed at: {datetime.now().isoformat()}")
    logger.info("=" * 60)


# ============================================================
# Entry Point
# ============================================================
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    # Run the batch prediction pipeline
    run_batch_prediction(args)

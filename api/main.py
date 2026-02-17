# ============================================================
# api/main.py
# FastAPI application entry point.
# Sets up the API server with all routes, middleware,
# model loading, and error handling.
# ============================================================

import sys                                         # System-specific parameters
from pathlib import Path                           # Object-oriented paths

# Add the project root to the Python path so we can import src modules
# This allows imports like: from src.utils.helpers import load_config
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException         # FastAPI framework and HTTP errors
from fastapi.middleware.cors import CORSMiddleware  # CORS middleware for cross-origin requests
from contextlib import asynccontextmanager         # Async context manager for lifespan events
import pandas as pd                                # Pandas for data manipulation
import numpy as np                                 # NumPy for numerical operations
from loguru import logger                          # Structured logging
import joblib                                      # Model deserialization

from api.schemas import (                          # Import all request/response schemas
    CustomerFeatures,                              # Single customer input
    PredictionResponse,                            # Single prediction output
    BatchPredictionRequest,                        # Batch input
    BatchPredictionResponse,                       # Batch output
    HealthResponse,                                # Health check output
    WhatIfRequest,                                 # What-if input
    WhatIfResponse,                                # What-if output
    RiskLevel,                                     # Risk level enum
)
from src.utils.helpers import load_config, load_model  # Config and model loading utilities
from src.features.engineer import FeatureEngineer  # Feature engineering module


# ============================================================
# Global variables for loaded model and config
# ============================================================
# These are populated at startup and used by all endpoints
app_state = {
    "model": None,                                 # The loaded ML model
    "config": None,                                # The loaded config
    "preprocessor": None,                          # The fitted preprocessor
    "engineer": None,                              # The feature engineer
    "selected_features": None,                     # Selected feature names after selection
    "feature_names": None,                         # Expected feature names
    "threshold": 0.5,                              # Decision threshold
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler: runs on startup and shutdown.

    On startup: loads the config, model, and preprocessor.
    On shutdown: cleans up resources.
    """
    # --- STARTUP ---
    logger.info("Starting API server...")

    try:
        # Load the project configuration
        config = load_config()
        app_state["config"] = config
        logger.info("Configuration loaded")

        # Load the trained model
        model_path = config["api"]["model_path"]
        if Path(model_path).exists():
            app_state["model"] = load_model(model_path)
            logger.info(f"Model loaded from: {model_path}")
        else:
            logger.warning(f"Model file not found at: {model_path}. API will start but predictions disabled.")

        # Initialize the feature engineer
        app_state["engineer"] = FeatureEngineer(config)
        logger.info("Feature engineer initialized")

        # Load the preprocessor if saved
        preprocessor_path = Path("models/preprocessor.joblib")
        if preprocessor_path.exists():
            app_state["preprocessor"] = joblib.load(preprocessor_path)
            logger.info("Preprocessor loaded")

        # Load selected feature names if saved
        selected_features_path = Path("models/selected_features.joblib")
        if selected_features_path.exists():
            app_state["selected_features"] = joblib.load(selected_features_path)
            logger.info(f"Selected features loaded: {len(app_state['selected_features'])} features")

        # Load feature names if saved
        feature_names_path = Path("models/feature_names.joblib")
        if feature_names_path.exists():
            app_state["feature_names"] = joblib.load(feature_names_path)
            logger.info(f"Feature names loaded: {len(app_state['feature_names'])} features")

        # Load the optimal threshold if saved
        threshold_path = Path("models/optimal_threshold.joblib")
        if threshold_path.exists():
            app_state["threshold"] = joblib.load(threshold_path)
            logger.info(f"Optimal threshold loaded: {app_state['threshold']}")

    except Exception as e:
        # Log the error but don't crash the server
        logger.error(f"Startup error: {e}")

    # Yield control to the application (it runs here)
    yield

    # --- SHUTDOWN ---
    logger.info("Shutting down API server...")


# ============================================================
# Create the FastAPI application instance
# ============================================================
app = FastAPI(
    title="Telco Churn Prediction API",            # API title shown in docs
    description=(                                   # API description for documentation
        "Advanced churn prediction API with explainability, "
        "what-if analysis, and retention action recommendations. "
        "Predicts customer churn probability and provides actionable insights."
    ),
    version="1.0.0",                               # API version
    lifespan=lifespan,                             # Startup/shutdown handler
)

# ============================================================
# CORS Middleware
# ============================================================
# Allow cross-origin requests (needed for Streamlit dashboard)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],                           # Allow all origins (restrict in production)
    allow_credentials=True,                        # Allow cookies
    allow_methods=["*"],                           # Allow all HTTP methods
    allow_headers=["*"],                           # Allow all headers
)


# ============================================================
# Helper Functions
# ============================================================

def customer_to_dataframe(customer: CustomerFeatures) -> pd.DataFrame:
    """
    Convert a CustomerFeatures Pydantic model to a pandas DataFrame.

    Parameters
    ----------
    customer : CustomerFeatures
        The validated customer input data.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with the customer's features.
    """
    # Convert the Pydantic model to a dictionary
    data = customer.model_dump()
    # Create a single-row DataFrame
    df = pd.DataFrame([data])
    # Return the DataFrame
    return df


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """
    Run the full feature pipeline: engineer -> preprocess -> select.

    Parameters
    ----------
    df : pd.DataFrame
        Raw customer DataFrame (one or more rows).

    Returns
    -------
    np.ndarray
        Model-ready feature array.
    """
    # Step 1: Feature engineering (add derived features)
    if app_state["engineer"] is not None:
        df = app_state["engineer"].engineer_all_features(df)

    # Step 2: Preprocessing (encode, scale, impute)
    if app_state["preprocessor"] is not None:
        features = app_state["preprocessor"].transform(df)
    else:
        features = df.values

    # Step 3: Feature selection (select only important features)
    if app_state["selected_features"] is not None:
        # Build feature names that match how training saved them
        # Training uses _get_feature_names(), or falls back to generic names
        try:
            feature_names = app_state["preprocessor"]._get_feature_names()
        except Exception:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]

        # Check if any selected features match; if not, use generic names
        overlap = [f for f in app_state["selected_features"] if f in feature_names]
        if len(overlap) == 0:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]

        features_df = pd.DataFrame(features, columns=feature_names)
        available = [f for f in app_state["selected_features"] if f in features_df.columns]
        if available:
            features = features_df[available].values
        else:
            logger.warning(f"No matching features found. Selected: {app_state['selected_features'][:5]}, Available: {feature_names[:5]}")

    return features


def classify_risk(probability: float) -> RiskLevel:
    """
    Classify a churn probability into a risk level.

    Parameters
    ----------
    probability : float
        Churn probability between 0 and 1.

    Returns
    -------
    RiskLevel
        The corresponding risk level enum value.
    """
    # Apply risk level thresholds
    if probability >= 0.7:
        return RiskLevel.HIGH                      # 70%+ = HIGH risk
    elif probability >= 0.4:
        return RiskLevel.MEDIUM                    # 40-70% = MEDIUM risk
    else:
        return RiskLevel.LOW                       # <40% = LOW risk


# ============================================================
# API Endpoints
# ============================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.

    Returns the API status, whether the model is loaded,
    and the API version. Used by monitoring systems and
    load balancers to verify the service is running.
    """
    return HealthResponse(
        status="healthy",                          # API is running
        model_loaded=app_state["model"] is not None,  # Is the model available?
        version="1.0.0",                           # API version
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_churn(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.

    Takes customer features as input and returns:
    - Churn probability (0.0 to 1.0)
    - Risk level (LOW, MEDIUM, HIGH)
    - Binary churn prediction (based on optimal threshold)
    - Top churn drivers (if explainer is available)
    - Recommended retention actions

    Raises 503 if the model is not loaded.
    """
    # Check that the model is loaded
    if app_state["model"] is None:
        # Return 503 Service Unavailable if model isn't ready
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train and deploy a model first.",
        )

    try:
        # Convert customer data to DataFrame
        df = customer_to_dataframe(customer)

        # Run full feature pipeline (engineer -> preprocess -> select)
        features = prepare_features(df)

        # Get churn probability from the model
        # predict_proba returns [[P(not churn), P(churn)]]
        probabilities = app_state["model"].predict_proba(features)
        # Extract the churn probability (positive class, index 1)
        churn_prob = float(probabilities[0][1])

        # Classify risk level
        risk = classify_risk(churn_prob)

        # Determine binary prediction using optimal threshold
        threshold = app_state["threshold"]
        will_churn = churn_prob >= threshold

        # Log the prediction
        logger.info(
            f"Prediction: P(churn)={churn_prob:.3f}, "
            f"Risk={risk.value}, Threshold={threshold}"
        )

        # Build and return the response
        return PredictionResponse(
            churn_probability=round(churn_prob, 4),
            risk_level=risk,
            threshold_used=threshold,
            will_churn=will_churn,
            top_drivers=[],                        # Populated when explainer is integrated
            recommended_actions=[],                # Populated when recommender is integrated
        )

    except Exception as e:
        # Log the error and return 500
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn for multiple customers in a single request.

    More efficient than calling /predict multiple times.
    Returns predictions for all customers plus a summary.
    """
    # Check that the model is loaded
    if app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # Convert all customers to DataFrames
        dfs = [customer_to_dataframe(c) for c in request.customers]
        # Concatenate into a single DataFrame
        batch_df = pd.concat(dfs, ignore_index=True)

        # Run full feature pipeline (engineer -> preprocess -> select)
        features = prepare_features(batch_df)

        # Get predictions for all customers at once
        probabilities = app_state["model"].predict_proba(features)
        # Extract churn probabilities (column index 1)
        churn_probs = probabilities[:, 1]

        # Build individual prediction responses
        predictions = []
        for prob in churn_probs:
            prob = float(prob)
            risk = classify_risk(prob)
            predictions.append(PredictionResponse(
                churn_probability=round(prob, 4),
                risk_level=risk,
                threshold_used=app_state["threshold"],
                will_churn=prob >= app_state["threshold"],
                top_drivers=[],
                recommended_actions=[],
            ))

        # Build the batch summary
        summary = {
            "total_customers": len(predictions),
            "avg_churn_probability": round(float(np.mean(churn_probs)), 4),
            "high_risk_count": sum(1 for p in churn_probs if p >= 0.7),
            "medium_risk_count": sum(1 for p in churn_probs if 0.4 <= p < 0.7),
            "low_risk_count": sum(1 for p in churn_probs if p < 0.4),
            "predicted_churners": sum(1 for p in churn_probs if p >= app_state["threshold"]),
        }

        # Log the batch summary
        logger.info(f"Batch prediction: {summary}")

        # Return the batch response
        return BatchPredictionResponse(
            predictions=predictions,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/whatif", response_model=WhatIfResponse, tags=["Analysis"])
async def what_if_analysis(request: WhatIfRequest):
    """
    Perform what-if analysis: how do proposed changes affect churn risk?

    Example: What if the customer switches to a yearly contract?
    """
    # Check that the model is loaded
    if app_state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        # Get original prediction
        original_df = customer_to_dataframe(request.customer)

        # Run full feature pipeline on original data
        original_features = prepare_features(original_df)
        original_prob = float(app_state["model"].predict_proba(original_features)[0][1])

        # Apply proposed changes to a copy of the raw data
        modified_df = customer_to_dataframe(request.customer)
        for feature, new_value in request.changes.items():
            if feature in modified_df.columns:
                modified_df[feature] = new_value

        # Run full feature pipeline on modified data
        modified_features = prepare_features(modified_df)
        new_prob = float(app_state["model"].predict_proba(modified_features)[0][1])

        # Calculate change
        delta = new_prob - original_prob

        # Log the analysis
        logger.info(f"What-if: {original_prob:.3f} → {new_prob:.3f} (Δ={delta:+.3f})")

        # Return the response
        return WhatIfResponse(
            original_probability=round(original_prob, 4),
            new_probability=round(new_prob, 4),
            probability_change=round(delta, 4),
            risk_reduced=delta < 0,
            changes_applied=request.changes,
        )

    except Exception as e:
        logger.error(f"What-if error: {e}")
        raise HTTPException(status_code=500, detail=f"What-if analysis failed: {str(e)}")


# ============================================================
# Run the server (when executed directly)
# ============================================================
if __name__ == "__main__":
    import uvicorn                                 # ASGI server
    # Run the FastAPI app with uvicorn
    uvicorn.run(
        "api.main:app",                            # Application path
        host="0.0.0.0",                            # Bind to all interfaces
        port=8000,                                 # Port number
        reload=True,                               # Auto-reload on code changes (dev only)
    )

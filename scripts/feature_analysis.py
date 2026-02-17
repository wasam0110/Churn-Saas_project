# ============================================================
# scripts/feature_analysis.py
# SHAP-based feature analysis script.
# Identifies the most impactful features, visualizes SHAP
# values, and suggests new feature engineering opportunities.
# ============================================================

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from loguru import logger
from datetime import datetime

from src.utils.helpers import load_config, setup_logging, load_model, ensure_directory
from src.features.engineer import FeatureEngineer


def run_feature_analysis():
    """Run complete SHAP-based feature analysis."""
    config = load_config()
    setup_logging(config.get("logging", {}).get("dir", "logs"))

    logger.info("=" * 60)
    logger.info("FEATURE ANALYSIS PIPELINE")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("=" * 60)

    # ----------------------------------------------------------
    # Step 1: Load Resources
    # ----------------------------------------------------------
    logger.info("Step 1: Loading model and data...")

    model = load_model("models/best_model.joblib")
    preprocessor = joblib.load("models/preprocessor.joblib")
    selected_features = joblib.load("models/selected_features.joblib")

    data_path = config.get("data", {}).get("raw_path", "data/raw/telco_churn.csv")
    df = pd.read_csv(data_path)
    logger.info(f"Data loaded: {df.shape[0]} rows")

    # Feature engineering
    engineer = FeatureEngineer(config)
    df_eng = engineer.engineer_all_features(df)

    # Prepare target
    target_col = config.get("features", {}).get("target", "Churn")
    if df_eng[target_col].dtype == object:
        df_eng[target_col] = (df_eng[target_col] == "Yes").astype(int)
    y = df_eng[target_col].values

    # Remove target and ID
    id_col = config.get("data", {}).get("id_column", "customerID")
    df_features = df_eng.drop(columns=[target_col, id_col], errors="ignore")

    # Preprocess
    X = preprocessor.transform(df_features)
    try:
        feature_names = preprocessor._get_feature_names()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    # If selected features were saved with generic names (feature_0, ...),
    # but the preprocessor returns real names (num__/cat__), fall back.
    if not any(f in feature_names for f in selected_features):
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    X_df = pd.DataFrame(X, columns=feature_names)
    available = [f for f in selected_features if f in X_df.columns]
    X_selected = X_df[available]

    logger.info(f"Features prepared: {X_selected.shape[1]} selected features")

    # ----------------------------------------------------------
    # Step 2: SHAP Analysis
    # ----------------------------------------------------------
    logger.info("Step 2: Computing SHAP values...")

    try:
        import shap

        # Use a sample for performance
        sample_size = min(1000, len(X_selected))
        X_sample = X_selected.iloc[:sample_size]

        # Choose explainer based on model type
        model_type = type(model).__name__
        logger.info(f"Model type: {model_type}")

        if hasattr(model, "feature_importances_"):
            explainer = shap.TreeExplainer(model)
        else:
            # Linear or general model
            explainer = shap.LinearExplainer(model, X_sample)

        shap_values = explainer.shap_values(X_sample)

        # Handle binary classification output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class

        logger.info(f"SHAP values computed: shape {shap_values.shape}")

    except Exception as e:
        logger.warning(f"SHAP computation failed: {e}")
        logger.info("Falling back to permutation importance...")

        from sklearn.inspection import permutation_importance
        result = permutation_importance(
            model, X_selected.values, y[:len(X_selected)],
            n_repeats=10, random_state=42, scoring="roc_auc"
        )
        shap_values = None
        perm_importance = pd.DataFrame({
            "Feature": available,
            "Importance": result.importances_mean,
            "Std": result.importances_std,
        }).sort_values("Importance", ascending=False)

    # ----------------------------------------------------------
    # Step 3: Generate Reports
    # ----------------------------------------------------------
    logger.info("Step 3: Generating analysis reports...")
    output_dir = ensure_directory("reports/feature_analysis")

    if shap_values is not None:
        # Global feature importance from SHAP
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        importance_df = pd.DataFrame({
            "Feature": available,
            "Mean |SHAP|": mean_abs_shap,
        }).sort_values("Mean |SHAP|", ascending=False)

        logger.info("Top 15 features by SHAP importance:")
        for _, row in importance_df.head(15).iterrows():
            logger.info(f"  {row['Feature']}: {row['Mean |SHAP|']:.4f}")

        # Plot 1: SHAP Summary Bar Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        top_n = min(20, len(importance_df))
        top = importance_df.head(top_n).iloc[::-1]
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, top_n))
        ax.barh(range(top_n), top["Mean |SHAP|"].values, color=colors, edgecolor="#333")
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top["Feature"].values, fontsize=10)
        ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
        ax.set_title("Feature Importance (SHAP)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        plot_path = str(output_dir / "shap_importance.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"SHAP importance plot saved: {plot_path}")

        # Plot 2: SHAP Beeswarm (top 15 features)
        try:
            fig2, ax2 = plt.subplots(figsize=(12, 8))
            shap.summary_plot(
                shap_values[:, :top_n],
                X_sample.iloc[:, :top_n],
                show=False,
                plot_size=(12, 8),
            )
            plt.tight_layout()
            plot_path2 = str(output_dir / "shap_beeswarm.png")
            plt.savefig(plot_path2, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"SHAP beeswarm plot saved: {plot_path2}")
        except Exception as e:
            logger.warning(f"Beeswarm plot failed: {e}")

        # Save importance data
        importance_df.to_csv(str(output_dir / "feature_importance.csv"), index=False)

    else:
        # Permutation importance fallback
        logger.info("Top 15 features by permutation importance:")
        for _, row in perm_importance.head(15).iterrows():
            logger.info(f"  {row['Feature']}: {row['Importance']:.4f} (+/- {row['Std']:.4f})")

        fig, ax = plt.subplots(figsize=(12, 8))
        top_n = min(20, len(perm_importance))
        top = perm_importance.head(top_n).iloc[::-1]
        ax.barh(range(top_n), top["Importance"].values, xerr=top["Std"].values,
                color="#58a6ff", edgecolor="#333", capsize=3)
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(top["Feature"].values, fontsize=10)
        ax.set_xlabel("Permutation Importance", fontsize=12)
        ax.set_title("Feature Importance (Permutation)", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        plot_path = str(output_dir / "permutation_importance.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Permutation importance plot saved: {plot_path}")

        perm_importance.to_csv(str(output_dir / "feature_importance.csv"), index=False)

    # ----------------------------------------------------------
    # Step 4: Feature Recommendations
    # ----------------------------------------------------------
    logger.info("Step 4: Feature engineering recommendations...")

    recommendations = [
        {
            "recommendation": "Interaction features between Contract and tenure",
            "rationale": "Contract type and tenure are both top predictors; their interaction may capture retention patterns.",
            "example": "tenure * (1 if Contract == 'Month-to-month' else 0)",
        },
        {
            "recommendation": "MonthlyCharges relative to service count",
            "rationale": "Customers paying high charges relative to services used may have higher churn risk.",
            "example": "MonthlyCharges / (num_services + 1)",
        },
        {
            "recommendation": "Tenure-based segments",
            "rationale": "New customers (< 6 months) and long-tenure customers show different churn patterns.",
            "example": "is_new_customer = (tenure <= 6).astype(int)",
        },
        {
            "recommendation": "Payment method risk score",
            "rationale": "Electronic check users churn at much higher rates. Combine with billing type.",
            "example": "high_risk_payment = (PaymentMethod == 'Electronic check') & (PaperlessBilling == 'Yes')",
        },
        {
            "recommendation": "Service bundle indicator",
            "rationale": "Customers bundling multiple services may be more sticky.",
            "example": "has_security_bundle = (OnlineSecurity == 'Yes') & (DeviceProtection == 'Yes')",
        },
    ]

    logger.info("Feature engineering suggestions:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"  {i}. {rec['recommendation']}")
        logger.info(f"     Rationale: {rec['rationale']}")
        logger.info(f"     Example: {rec['example']}")

    # Save recommendations
    rec_df = pd.DataFrame(recommendations)
    rec_df.to_csv(str(output_dir / "feature_recommendations.csv"), index=False)
    logger.info(f"Recommendations saved to: {output_dir / 'feature_recommendations.csv'}")

    logger.info("=" * 60)
    logger.info("FEATURE ANALYSIS COMPLETE")
    logger.info(f"Reports saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_feature_analysis()

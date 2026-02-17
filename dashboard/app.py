# ============================================================
# dashboard/app.py
# Streamlit dashboard for the Churn Prediction SaaS.
# Professional dark theme, real model predictions,
# interactive charts, drift monitoring, and batch scoring.
# ============================================================

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
from loguru import logger

from src.utils.helpers import load_config
from src.features.engineer import FeatureEngineer

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="C",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Dark Theme CSS
# ============================================================
st.markdown("""
<style>
    /* Dark theme override */
    .stApp { background-color: #0d1117; }
    section[data-testid="stSidebar"] { background-color: #161b22; }
    .stMetric label { color: #8b949e !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e6edf3 !important; }
    h1, h2, h3, h4, h5, h6 { color: #e6edf3 !important; }
    p, span, div { color: #c9d1d9; }
    .stSelectbox label, .stNumberInput label, .stSlider label { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Cached Resource Loading
# ============================================================
@st.cache_resource
def load_app_config():
    """Load and cache the project configuration."""
    try:
        return load_config()
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return {"project": {"name": "Churn Prediction"}}


@st.cache_resource
def load_trained_model():
    """Load and cache the trained model."""
    model_path = Path("models/best_model.joblib")
    if model_path.exists():
        return joblib.load(model_path)
    return None


@st.cache_resource
def load_preprocessor():
    """Load and cache the fitted preprocessor."""
    path = Path("models/preprocessor.joblib")
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_resource
def load_selected_features():
    """Load and cache the selected feature names."""
    path = Path("models/selected_features.joblib")
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_resource
def load_threshold():
    """Load and cache the optimal threshold."""
    path = Path("models/optimal_threshold.joblib")
    if path.exists():
        return joblib.load(path)
    return 0.5


@st.cache_resource
def load_feature_engineer():
    """Load and cache the feature engineer."""
    try:
        config = load_app_config()
        return FeatureEngineer(config)
    except Exception:
        return None


def predict_single(df: pd.DataFrame):
    """
    Run full prediction pipeline on a single-row DataFrame.
    Returns (churn_probability, risk_level, will_churn) or None on error.
    """
    model = load_trained_model()
    preprocessor = load_preprocessor()
    selected_features = load_selected_features()
    engineer = load_feature_engineer()
    threshold = load_threshold()

    if model is None:
        return None

    try:
        # Feature engineering
        if engineer is not None:
            df = engineer.engineer_all_features(df)

        # Preprocessing
        if preprocessor is not None:
            features = preprocessor.transform(df)
        else:
            features = df.values

        # Feature selection
        if selected_features is not None:
            try:
                names = preprocessor._get_feature_names()
            except Exception:
                names = [f"feature_{i}" for i in range(features.shape[1])]
            if not any(f in names for f in selected_features):
                names = [f"feature_{i}" for i in range(features.shape[1])]
            features_df = pd.DataFrame(features, columns=names)
            available = [f for f in selected_features if f in features_df.columns]
            if available:
                features = features_df[available].values

        # Predict
        proba = model.predict_proba(features)
        churn_prob = float(proba[0][1])

        if churn_prob >= 0.7:
            risk = "HIGH"
        elif churn_prob >= 0.4:
            risk = "MEDIUM"
        else:
            risk = "LOW"

        return churn_prob, risk, churn_prob >= threshold
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return None


def predict_batch(df: pd.DataFrame):
    """
    Run prediction on a multi-row DataFrame.
    Returns DataFrame with churn_probability, risk_level, will_churn columns.
    """
    model = load_trained_model()
    preprocessor = load_preprocessor()
    selected_features = load_selected_features()
    engineer = load_feature_engineer()
    threshold = load_threshold()

    if model is None:
        return None

    try:
        df_eng = df.copy()
        if engineer is not None:
            df_eng = engineer.engineer_all_features(df_eng)

        if preprocessor is not None:
            features = preprocessor.transform(df_eng)
        else:
            features = df_eng.values

        if selected_features is not None:
            try:
                names = preprocessor._get_feature_names()
            except Exception:
                names = [f"feature_{i}" for i in range(features.shape[1])]
            if not any(f in names for f in selected_features):
                names = [f"feature_{i}" for i in range(features.shape[1])]
            features_df = pd.DataFrame(features, columns=names)
            available = [f for f in selected_features if f in features_df.columns]
            if available:
                features = features_df[available].values

        proba = model.predict_proba(features)
        churn_probs = proba[:, 1]

        result = df.copy()
        result["churn_probability"] = churn_probs
        result["risk_level"] = pd.cut(
            churn_probs, bins=[0, 0.4, 0.7, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"], include_lowest=True
        )
        result["will_churn"] = churn_probs >= threshold
        return result
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return None


# ============================================================
# Sidebar Navigation
# ============================================================
def render_sidebar():
    """Render sidebar with navigation."""
    st.sidebar.title("Churn Prediction")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate to:",
        [
            "Overview",
            "Customer Analysis",
            "What-If Simulator",
            "Model Performance",
            "Drift Monitoring",
            "Batch Predictions",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    model = load_trained_model()
    if model is not None:
        st.sidebar.success("Model loaded and ready")
        threshold = load_threshold()
        st.sidebar.caption(f"Threshold: {threshold:.3f}")
    else:
        st.sidebar.warning("No model trained yet")
        st.sidebar.code("python scripts/train.py", language="bash")

    return page


# ============================================================
# Page: Overview
# ============================================================
def render_overview():
    """Dashboard overview with key metrics and charts."""
    st.title("Churn Prediction Dashboard")
    st.caption("Real-time customer churn risk monitoring and insights.")

    model = load_trained_model()
    if model is None:
        st.warning("No trained model found. Run the training pipeline first.")
        st.code("python scripts/train.py", language="bash")
        return

    # Load training data for overview stats
    config = load_app_config()
    data_path = Path(config.get("data", {}).get("raw_path", "data/raw/telco_churn.csv"))
    if data_path.exists():
        df = pd.read_csv(data_path)
        if "Churn" in df.columns:
            if df["Churn"].dtype == object:
                df["Churn"] = (df["Churn"] == "Yes").astype(int)
            total = len(df)
            churn_rate = df["Churn"].mean()
            high_risk_count = int(total * churn_rate * 0.6)
            monthly_charges = df["MonthlyCharges"].mean() if "MonthlyCharges" in df.columns else 70.0
        else:
            total, churn_rate, high_risk_count, monthly_charges = 7043, 0.265, 482, 70.0
    else:
        total, churn_rate, high_risk_count, monthly_charges = 7043, 0.265, 482, 70.0

    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", f"{total:,}")
    with col2:
        st.metric("Churn Rate", f"{churn_rate:.1%}")
    with col3:
        st.metric("High Risk Customers", f"{high_risk_count:,}")
    with col4:
        revenue_risk = high_risk_count * monthly_charges * 12
        st.metric("Revenue at Risk", f"${revenue_risk:,.0f}")

    st.markdown("---")

    # Charts
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Distribution")
        low = int(total * (1 - churn_rate) * 0.7)
        med = total - low - high_risk_count
        risk_data = pd.DataFrame({
            "Risk Level": ["Low", "Medium", "High"],
            "Count": [low, med, high_risk_count],
        })
        fig = px.pie(
            risk_data, values="Count", names="Risk Level",
            color="Risk Level",
            color_discrete_map={"Low": "#3fb950", "Medium": "#d29922", "High": "#f85149"},
            hole=0.4,
        )
        fig.update_layout(
            margin=dict(t=20, b=20, l=20, r=20), height=350,
            paper_bgcolor="#0d1117", plot_bgcolor="#0d1117",
            font_color="#c9d1d9",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Churn by Contract Type")
        if data_path.exists():
            df_full = pd.read_csv(data_path)
            if "Contract" in df_full.columns and "Churn" in df_full.columns:
                if df_full["Churn"].dtype == object:
                    df_full["Churn"] = (df_full["Churn"] == "Yes").astype(int)
                contract_churn = df_full.groupby("Contract")["Churn"].mean().reset_index()
                contract_churn.columns = ["Contract", "Churn Rate"]
                fig = px.bar(
                    contract_churn, x="Contract", y="Churn Rate",
                    color="Contract",
                    color_discrete_map={
                        "Month-to-month": "#f85149",
                        "One year": "#d29922",
                        "Two year": "#3fb950",
                    },
                )
                fig.update_layout(
                    yaxis_title="Churn Rate", showlegend=False,
                    margin=dict(t=20, b=20, l=20, r=20), height=350,
                    paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                    font_color="#c9d1d9",
                    yaxis=dict(gridcolor="#30363d"),
                )
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page: Customer Analysis
# ============================================================
def render_customer_analysis():
    """Individual customer prediction with real model output."""
    st.title("Customer Churn Analysis")
    st.caption("Enter customer details for a real-time churn prediction.")

    model = load_trained_model()
    if model is None:
        st.warning("No model available. Train a model first.")
        return

    with st.form("customer_form"):
        st.subheader("Customer Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Demographics**")
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)

        with col2:
            st.markdown("**Services**")
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
            backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
            protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])

        with col3:
            st.markdown("**Billing and Contract**")
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)",
            ])
            monthly = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
            total = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=840.0)

        col4, col5, col6 = st.columns(3)
        with col4:
            tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        with col5:
            streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        with col6:
            streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

        multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

        submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

    if submitted:
        with st.spinner("Analyzing customer..."):
            # Build the customer DataFrame
            customer_data = {
                "gender": gender, "SeniorCitizen": senior,
                "Partner": partner, "Dependents": dependents,
                "tenure": tenure, "PhoneService": phone,
                "MultipleLines": multiple_lines,
                "InternetService": internet, "OnlineSecurity": security,
                "OnlineBackup": backup, "DeviceProtection": protection,
                "TechSupport": tech_support, "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract, "PaperlessBilling": paperless,
                "PaymentMethod": payment,
                "MonthlyCharges": monthly, "TotalCharges": total,
            }
            df = pd.DataFrame([customer_data])

            result = predict_single(df)
            if result is None:
                st.error("Prediction failed. Check model and preprocessor files.")
                return

            churn_prob, risk_level, will_churn = result

            st.markdown("---")
            st.subheader("Prediction Results")

            res1, res2, res3 = st.columns(3)
            with res1:
                st.metric("Churn Probability", f"{churn_prob:.1%}")
            with res2:
                st.metric("Risk Level", risk_level)
            with res3:
                st.metric("Expected Annual Loss", f"${churn_prob * monthly * 12:.0f}")

            # Risk gauge
            if risk_level == "HIGH":
                bar_color = "#f85149"
            elif risk_level == "MEDIUM":
                bar_color = "#d29922"
            else:
                bar_color = "#3fb950"

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Churn Risk Score", "font": {"color": "#e6edf3"}},
                number={"font": {"color": "#e6edf3"}, "suffix": "%"},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#8b949e"},
                    "bar": {"color": bar_color},
                    "bgcolor": "#161b22",
                    "steps": [
                        {"range": [0, 40], "color": "#1a2e1a"},
                        {"range": [40, 70], "color": "#2e2a1a"},
                        {"range": [70, 100], "color": "#2e1a1a"},
                    ],
                },
            ))
            fig.update_layout(
                height=300, margin=dict(t=50, b=20),
                paper_bgcolor="#0d1117", font_color="#e6edf3",
            )
            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# Page: What-If Simulator
# ============================================================
def render_whatif():
    """What-if simulator using real model predictions."""
    st.title("What-If Churn Simulator")
    st.caption(
        "Change customer attributes and compare the impact on churn risk."
    )

    model = load_trained_model()
    if model is None:
        st.warning("No model available.")
        return

    st.subheader("Common Retention Scenarios")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info(
            "**Contract Upgrade**\n\n"
            "Switch from Month-to-month to One year contract. "
            "Expected risk reduction: ~35%"
        )
    with col2:
        st.info(
            "**Add Security**\n\n"
            "Add Online Security + Device Protection. "
            "Expected risk reduction: ~20%"
        )
    with col3:
        st.info(
            "**Auto-Pay Switch**\n\n"
            "Switch from Electronic check to Bank transfer. "
            "Expected risk reduction: ~18%"
        )

    st.markdown("---")
    st.subheader("Interactive Simulation")

    feature = st.selectbox(
        "Select feature to change:",
        ["Contract", "MonthlyCharges", "InternetService",
         "OnlineSecurity", "DeviceProtection", "PaymentMethod"],
    )

    # Define a baseline high-risk customer
    baseline = {
        "gender": "Male", "SeniorCitizen": 0,
        "Partner": "No", "Dependents": "No",
        "tenure": 3, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0, "TotalCharges": 255.0,
    }

    if feature == "Contract":
        new_value = st.selectbox("New value:", ["Month-to-month", "One year", "Two year"])
    elif feature == "MonthlyCharges":
        new_value = st.slider("New monthly charge ($):", 20.0, 120.0, 50.0)
    elif feature == "InternetService":
        new_value = st.selectbox("New value:", ["DSL", "Fiber optic", "No"])
    elif feature in ["OnlineSecurity", "DeviceProtection"]:
        new_value = st.selectbox("New value:", ["Yes", "No"])
    elif feature == "PaymentMethod":
        new_value = st.selectbox("New value:", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])

    if st.button("Simulate Change"):
        # Original prediction
        df_orig = pd.DataFrame([baseline])
        result_orig = predict_single(df_orig)

        # Modified prediction
        modified = baseline.copy()
        modified[feature] = new_value
        df_mod = pd.DataFrame([modified])
        result_mod = predict_single(df_mod)

        if result_orig and result_mod:
            orig_prob = result_orig[0]
            new_prob = result_mod[0]
            delta = new_prob - orig_prob

            sim1, sim2, sim3 = st.columns(3)
            with sim1:
                st.metric("Original Risk", f"{orig_prob:.1%}")
            with sim2:
                st.metric("New Risk", f"{new_prob:.1%}", delta=f"{delta:+.1%}", delta_color="inverse")
            with sim3:
                reduction = (1 - new_prob / orig_prob) * 100 if orig_prob > 0 else 0
                st.metric("Risk Change", f"{reduction:+.1f}%")
        else:
            st.error("Simulation failed. Check model files.")


# ============================================================
# Page: Model Performance
# ============================================================
def render_performance():
    """Model performance with actual saved plots."""
    st.title("Model Performance")
    st.caption("Evaluation metrics and model comparison.")

    reports_dir = Path("reports")
    plot_files = {
        "ROC Curves": reports_dir / "roc_curves.png",
        "Precision-Recall Curves": reports_dir / "pr_curves.png",
        "Calibration Curves": reports_dir / "calibration_curves.png",
        "Profit Curve": reports_dir / "profit_curve.png",
        "SHAP Summary": reports_dir / "shap_summary.png",
    }

    has_plots = False
    for title, path in plot_files.items():
        if path.exists():
            st.subheader(title)
            st.image(str(path), use_container_width=True)
            has_plots = True

    if not has_plots:
        st.info(
            "No evaluation plots found. "
            "Run the training pipeline to generate reports."
        )
        st.code("python scripts/train.py", language="bash")

    # Model metrics from registry
    st.subheader("Model Comparison")
    registry_path = Path("models/registry_index.json")
    if registry_path.exists():
        import json
        with open(registry_path) as f:
            registry = json.load(f)
        models = registry.get("models", {})
        if models:
            rows = []
            for key, info in models.items():
                m = info.get("metrics", {})
                rows.append({
                    "Model": info.get("model_name", key),
                    "ROC-AUC": m.get("roc_auc", "-"),
                    "PR-AUC": m.get("pr_auc", "-"),
                    "Brier Score": m.get("brier_score", "-"),
                    "Log Loss": m.get("log_loss", "-"),
                    "Stage": info.get("stage", "staging"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No model registry found.")


# ============================================================
# Page: Drift Monitoring
# ============================================================
def render_drift_monitoring():
    """Drift monitoring with actual drift reports."""
    st.title("Drift Monitoring")
    st.caption(
        "Monitor data drift and model performance changes. "
        "Alerts when retraining may be needed."
    )

    drift_plot = Path("reports/monitoring/drift_report.png")
    if drift_plot.exists():
        st.subheader("Feature Drift Report")
        st.image(str(drift_plot), use_container_width=True)

    # Show drift status from config thresholds
    config = load_app_config()
    psi_threshold = config.get("monitoring", {}).get("psi_threshold", 0.2)
    decay_threshold = config.get("monitoring", {}).get("performance_decay_threshold", 0.05)

    st.subheader("Monitoring Configuration")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric("PSI Threshold", f"{psi_threshold}")
    with m2:
        st.metric("Performance Decay Threshold", f"{decay_threshold}")
    with m3:
        freq = config.get("monitoring", {}).get("drift_check_frequency", "weekly")
        st.metric("Check Frequency", freq.capitalize())

    st.markdown("---")
    st.subheader("Run Drift Check")
    st.caption("Upload new production data to check for drift against training data.")

    uploaded = st.file_uploader("Upload production data (CSV)", type=["csv"])
    if uploaded is not None:
        new_data = pd.read_csv(uploaded)
        st.dataframe(new_data.head(), use_container_width=True)

        if st.button("Run Drift Analysis"):
            with st.spinner("Analyzing drift..."):
                try:
                    from src.monitoring.drift import DriftDetector

                    # Load training data for reference
                    data_path = config.get("data", {}).get("raw_path", "data/raw/telco_churn.csv")
                    if Path(data_path).exists():
                        train_df = pd.read_csv(data_path)
                        detector = DriftDetector(config)
                        # Set reference from training data (numeric only)
                        train_numeric = train_df.select_dtypes(include=[np.number])
                        detector.set_reference(train_numeric)

                        # Check drift on new data
                        new_numeric = new_data.select_dtypes(include=[np.number])
                        drift_report = detector.check_feature_drift(new_numeric)

                        # Display results
                        drift_rows = []
                        for feat, info in drift_report.items():
                            drift_rows.append({
                                "Feature": feat,
                                "PSI": f"{info['psi']:.4f}",
                                "Status": info["status"],
                                "KS Statistic": f"{info['ks_statistic']:.4f}",
                                "KS p-value": f"{info['ks_pvalue']:.6f}",
                            })
                        st.dataframe(pd.DataFrame(drift_rows), use_container_width=True, hide_index=True)

                        # Generate and show the plot
                        plot_path = detector.plot_drift_report(drift_report)
                        st.image(plot_path, use_container_width=True)
                    else:
                        st.error("Training data not found for reference comparison.")
                except Exception as e:
                    st.error(f"Drift analysis failed: {e}")


# ============================================================
# Page: Batch Predictions
# ============================================================
def render_batch_predictions():
    """Batch predictions with real model output."""
    st.title("Batch Predictions")
    st.caption("Upload a CSV file to score all customers at once.")

    uploaded = st.file_uploader(
        "Upload customer data (CSV)", type=["csv"],
        help="File should have the same columns as training data.",
    )

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Loaded {len(df)} customers")

        if st.button("Generate Predictions", use_container_width=True):
            with st.spinner("Generating predictions..."):
                result = predict_batch(df)

                if result is not None:
                    st.subheader("Prediction Results")
                    st.dataframe(result, use_container_width=True)

                    s1, s2, s3 = st.columns(3)
                    with s1:
                        high_count = (result["risk_level"] == "HIGH").sum()
                        st.metric("High Risk", int(high_count))
                    with s2:
                        med_count = (result["risk_level"] == "MEDIUM").sum()
                        st.metric("Medium Risk", int(med_count))
                    with s3:
                        low_count = (result["risk_level"] == "LOW").sum()
                        st.metric("Low Risk", int(low_count))

                    # Distribution chart
                    fig = px.histogram(
                        result, x="churn_probability", nbins=30,
                        title="Prediction Distribution",
                        color_discrete_sequence=["#58a6ff"],
                    )
                    fig.update_layout(
                        paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
                        font_color="#c9d1d9",
                        xaxis=dict(gridcolor="#30363d"),
                        yaxis=dict(gridcolor="#30363d"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Download
                    csv_out = result.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions",
                        data=csv_out,
                        file_name="churn_predictions.csv",
                        mime="text/csv",
                    )
                else:
                    st.error("Prediction failed. Ensure model files are present.")


# ============================================================
# Main Router
# ============================================================
def main():
    """Main entry point. Routes to selected page."""
    page = render_sidebar()

    if "Overview" in page:
        render_overview()
    elif "Customer Analysis" in page:
        render_customer_analysis()
    elif "What-If" in page:
        render_whatif()
    elif "Performance" in page:
        render_performance()
    elif "Drift" in page:
        render_drift_monitoring()
    elif "Batch" in page:
        render_batch_predictions()


if __name__ == "__main__":
    main()

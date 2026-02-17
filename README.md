# Telco Churn Prediction SaaS

Professional, end-to-end churn prediction project: training, serving, dashboarding, monitoring, A/B testing and retraining utilities.

## Quick summary

- Trains models to predict customer churn using a repeatable pipeline: feature engineering → preprocessing → feature selection → model training & evaluation.
- Serves real-time predictions and what‑if analysis with a FastAPI service (`/predict`, `/predict/batch`, `/whatif`).
- Interactive Streamlit dashboard for overview, customer analysis, what‑if simulation, model performance and drift monitoring.
- CLI scripts for batch prediction, feature analysis (SHAP), drift monitoring, A/B testing and retraining.

## Quick start (Windows, PowerShell)

1. Activate the project virtual environment:

```powershell
& .\.venv\Scripts\Activate.ps1
```

2. Install dependencies (if not already):

```powershell
python -m pip install -r requirements.txt
```

3. Run tests:

```powershell
python -m pytest tests/ -v
```

4. Start the API (FastAPI + Uvicorn):

```powershell
python -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

5. Start the dashboard (Streamlit):

```powershell
python -m streamlit run dashboard\app.py --server.port 8501 --server.address 127.0.0.1
```

6. Run monitoring, feature analysis, or other utilities:

```powershell
# Drift monitoring
python scripts/monitor.py --data data/raw/telco_churn.csv --output reports/monitoring

# Feature analysis (SHAP + recommendations)
python scripts/feature_analysis.py

# Batch scoring
python scripts/predict.py --input data/raw/batch_input.csv --output reports/predictions.csv

# A/B test (example)
python scripts/ab_test.py --champion models/best_model.joblib --challenger models/challenger.joblib --data data/raw/telco_churn.csv

# Retrain pipeline
python scripts/retrain.py --data data/raw/telco_churn.csv --auto-promote
```

## Important files & folders

- `api/` — FastAPI service and pydantic schemas.
- `dashboard/` — Streamlit dashboard app (`app.py`).
- `scripts/` — CLI utilities: `train.py`, `predict.py`, `monitor.py`, `feature_analysis.py`, `ab_test.py`, `retrain.py`.
- `src/` — Core modules: feature engineering (`src/features/engineer.py`), selection (`src/features/selector.py`), preprocessing (`src/data/preprocessor.py`), monitoring (`src/monitoring/drift.py`), and utilities (`src/utils/helpers.py`).
- `models/` — Persisted model artifacts: `best_model.joblib`, `preprocessor.joblib`, `selected_features.joblib`, `optimal_threshold.joblib`, and `registry_index.json`.
- `reports/` — Generated reports and visualizations (training results, monitoring, SHAP).
- `config/config.yaml` — Project configuration: paths, features, hyperparameters and thresholds.

## API endpoints (examples)

- `GET /health` — health check.
- `POST /predict` — single customer prediction (JSON body matching `api/schemas.py::CustomerFeatures`).
- `POST /predict/batch` — batch predictions (list of customers).
- `POST /whatif` — what-if simulation (baseline customer + changes).

## Notes & recommendations

- Always run commands from the activated `.venv` to ensure correct dependencies.
- The `selected_features` saved from training may use generic `feature_0..` names — scripts include robust fallbacks to avoid feature-name mismatches between preprocessor output and saved selections.
- Use the `reports/feature_analysis` output to guide new feature engineering; integrate high-impact suggestions into `src/features/engineer.py`.

## Development workflow

1. Edit code & tests.
2. Run `pytest` and fix failing tests.
3. Retrain with `scripts/train.py` if data or features change.
4. Promote model (manual or via `--auto-promote` in `scripts/retrain.py`).

## License & contact

Add your license and maintainer contact information here.

---

Generated/updated on: 2026-02-17

#  Churn Prediction SaaS Project

> **Advanced Telco Customer Churn Prediction Platform** — A production-grade, end-to-end machine learning system that predicts customer churn, explains predictions, recommends retention actions, and monitors model performance in real time.

---

##  Architecture Overview

```
ChurnPredictionSaasProject/
├── config/
│   └── config.yaml              # Master configuration (paths, hyperparams, thresholds)
├── src/                          # Core ML package
│   ├── data/
│   │   ├── loader.py            # Data ingestion & cleaning
│   │   ├── validator.py         # Schema & quality validation
│   │   └── preprocessor.py      # Encoding, scaling, train/test split
│   ├── features/
│   │   ├── engineer.py          # 30+ engineered features (tenure, charge, service, interaction)
│   │   └── selector.py          # Correlation, mutual info, tree-based feature selection
│   ├── models/
│   │   ├── trainer.py           # Multi-model training (LR, RF, XGBoost, LightGBM)
│   │   ├── evaluator.py         # ROC-AUC, PR-AUC, lift@k, expected profit, calibration plots
│   │   ├── explainer.py         # SHAP-based global & local explanations + what-if analysis
│   │   └── registry.py          # File-based model versioning & promotion
│   ├── monitoring/
│   │   └── drift.py             # PSI-based feature & prediction drift detection
│   ├── actions/
│   │   └── recommender.py       # ROI-optimized next-best-action retention recommendations
│   └── utils/
│       └── helpers.py           # Config loading, logging, model I/O, path utilities
├── api/
│   ├── main.py                  # FastAPI app (predict, batch, what-if, health endpoints)
│   └── schemas.py               # Pydantic v2 request/response models with validation
├── dashboard/
│   └── app.py                   # Streamlit interactive dashboard (6 pages)
├── scripts/
│   ├── train.py                 # Full training pipeline orchestrator
│   └── predict.py               # Batch prediction with CLI arguments
├── tests/
│   ├── test_data.py             # Data pipeline unit tests
│   ├── test_features.py         # Feature engineering tests
│   └── test_models.py           # Model training & evaluation tests
├── data/raw/                    # Raw input data (CSV)
├── data/processed/              # Processed outputs
├── models/                      # Saved models, preprocessors, thresholds
├── logs/                        # Application logs
├── reports/                     # Evaluation plots & drift reports
├── requirements.txt             # All Python dependencies
└── .gitignore                   # Comprehensive ignore rules
```

---

##  Key Features

###  Advanced Machine Learning

- **4 algorithms**: Logistic Regression, Random Forest, XGBoost, LightGBM
- **Probability calibration** via Isotonic Regression for reliable risk scores
- **Cost-sensitive threshold optimization** maximizing expected profit (not just accuracy)
- **Automated hyperparameter configuration** via YAML config

### 🔬 Feature Engineering (30+ Features)

- **Tenure features**: years, groups, bins, new customer flag, loyalty score
- **Charge features**: average per month, charge ratio, overpayment detection, spending tier
- **Service features**: total service count, has security, has streaming, digital engagement
- **Contract features**: risk score, auto-pay indicator, commitment level
- **Interaction features**: tenure × charges, contract × internet, senior × support

### 📊 Evaluation Beyond Accuracy

- ROC-AUC and PR-AUC with publication-quality curves
- **Lift@k%** — how many more churners you catch in the top k%
- **Expected profit curve** — find the threshold that maximizes business value
- **Calibration plots** — verify that a "60% risk" score is truly 60%
- Side-by-side model comparison dashboard

###  Explainability (SHAP)

- **Global importance** — which features matter most across all customers
- **Local explanations** — why THIS customer is predicted to churn
- **What-if analysis** — "what happens if we upgrade their contract?"
- SHAP summary and waterfall plots (auto-saved)

###  Drift Monitoring

- **Population Stability Index (PSI)** for every feature
- Prediction distribution drift detection
- Performance decay monitoring over time
- Configurable warning and alert thresholds

###  Next-Best-Action Engine

- **15-action catalog** across 5 categories (pricing, contract, service, payment, engagement)
- **ROI-based ranking** — each action estimates cost, success rate, and expected value
- **Personalized retention plans** with priority and implementation timeline
- **Customer prioritization** by expected revenue loss

###  REST API (FastAPI)

- `POST /predict` — single customer prediction with risk level
- `POST /predict/batch` — score hundreds of customers at once
- `POST /whatif` — simulate feature changes
- `GET /health` — model status and version info
- CORS-enabled, production-ready with async support

###  Interactive Dashboard (Streamlit)

- **Overview** — KPIs, risk distribution pie chart, monthly trend
- **Customer Analysis** — form-based individual prediction with gauge chart
- **What-If Simulator** — interactive feature manipulation
- **Model Performance** — evaluation plots and model comparison table
- **Drift Monitoring** — PSI trends and feature stability status
- **Batch Predictions** — CSV upload → download scored results

---

##  Quick Start

###  Clone the Repository

```bash
git clone https://github.com/wasam0110/Churn-Saas_project.git
cd Churn-Saas_project
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python scripts/train.py
```

> This will auto-generate synthetic data if no real data is present, train all 4 models, calibrate probabilities, optimize thresholds, compute SHAP explanations, and register the best model.

### 5. Launch the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs) for the interactive Swagger UI.

### 6. Launch the Dashboard

```bash
streamlit run dashboard/app.py --server.port 8501
```

Visit: [http://localhost:8501](http://localhost:8501)

### 7. Run Batch Predictions

```bash
python scripts/predict.py --input data/raw/telco_churn.csv --output data/processed/predictions.csv --actions
```

### 8. Run Tests

```bash
pytest tests/ -v
```

---

## 🔌 API Reference

### `POST /predict`

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 84.45,
  "TotalCharges": 1013.4
}
```

**Response:**

```json
{
  "churn_probability": 0.7234,
  "churn_prediction": true,
  "risk_level": "HIGH",
  "threshold_used": 0.45,
  "model_version": "xgboost_v1"
}
```

### `POST /predict/batch`

Send an array of customers and receive predictions for all of them.

### `POST /whatif`

Simulate changing one feature and see the impact on churn probability.

### `GET /health`

Check model status, version, and uptime.

---

##  Configuration

All settings are centralized in `config/config.yaml`:

| Section           | Parameters                                     |
| ----------------- | ---------------------------------------------- |
| **data**          | Paths, column names, target encoding           |
| **features**      | Tenure bins, rolling windows, service columns  |
| **preprocessing** | Test size, random state, scaling method        |
| **models**        | Algorithms, hyperparameters for LR/RF/XGB/LGBM |
| **calibration**   | Method (isotonic/sigmoid), CV folds            |
| **threshold**     | Retention cost ($50), customer value ($500)    |
| **monitoring**    | PSI thresholds (0.1 warning, 0.2 drift)        |
| **api**           | Host, port, CORS origins                       |
| **dashboard**     | Streamlit port, theme                          |

---

##  Testing

```bash
# Run all tests with verbose output
pytest tests/ -v

# Run specific test modules
pytest tests/test_data.py -v
pytest tests/test_features.py -v
pytest tests/test_models.py -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html
```

---

##  Tech Stack

| Category           | Technologies                                      |
| ------------------ | ------------------------------------------------- |
| **ML/DS**          | scikit-learn, XGBoost, LightGBM, imbalanced-learn |
| **Explainability** | SHAP, waterfall & summary plots                   |
| **API**            | FastAPI, Pydantic v2, Uvicorn                     |
| **Dashboard**      | Streamlit, Plotly                                 |
| **MLOps**          | MLflow, Evidently, custom model registry          |
| **Data**           | Pandas, NumPy, PyYAML                             |
| **Testing**        | pytest, pytest-cov                                |
| **Logging**        | Loguru                                            |

---

##  What Makes This Project Stand Out

1. **Business-Value Optimization** — Doesn't just maximize accuracy; optimizes the threshold for maximum expected profit using configurable retention costs and customer lifetime value.

2. **Calibrated Probabilities** — Uses isotonic regression calibration so that a "60% churn risk" truly means 60%.

3. **Full Explainability Stack** — SHAP-based global importance, per-customer explanations, and what-if analysis for actionable insights.

4. **Next-Best-Action Engine** — Goes beyond prediction to recommend specific retention actions ranked by ROI.

5. **Drift Monitoring** — PSI-based feature and prediction drift detection to know when the model needs retraining.

6. **Production Architecture** — FastAPI backend, Streamlit dashboard, model registry with versioning, and comprehensive logging.

7. **Every Line Commented** — Every single line of code has an explanatory comment for maximum readability and learning.

---

##  License

This project is for educational and portfolio purposes.

---

##  Author

Built with ❤️ as an advanced data science portfolio project.

## Getting started

1. Install dependencies (see project files for language-specific requirements).
2. Run preprocessing and training scripts to prepare the model.
3. Start the web service to serve predictions.

## Development

- Add your project files into this repository.
- Commit and push changes to the `main` branch.

## Remote

Repository remote: https://github.com/wasam0110/Churn-Saas_project.git

---

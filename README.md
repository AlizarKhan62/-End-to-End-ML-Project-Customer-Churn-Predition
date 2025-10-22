
# 💼 End-to-End Customer Churn Prediction System

> **An enterprise-level machine learning project that predicts telecom customer churn, visualizes insights with Power BI, tracks experiments with MLflow, and deploys an interactive web app using Streamlit Cloud.**

---

## 🧠 Project Overview

This project demonstrates the **complete lifecycle of an ML solution** — from **data ingestion** to **cloud deployment**, integrating **MLflow, Docker, CI/CD pipelines**, and an **interactive Streamlit dashboard**.  

It uses the popular **Telco Customer Churn dataset**, where the goal is to predict whether a customer will leave (churn) or stay, based on demographic and service-related attributes.

---

## 🚀 Key Features

| Category | Description |
|-----------|--------------|
| 🧹 **Data Pipeline** | End-to-end cleaning, encoding, feature scaling, and transformation |
| 🧮 **Machine Learning** | Model training using `XGBoost` and `RandomForestClassifier` |
| 📊 **Power BI Dashboard** | Interactive KPIs and churn segmentation dashboard (two-page setup) |
| 🧠 **Explainability** | SHAP, ROC, Precision-Recall curves, feature importance |
| 🔍 **Experiment Tracking** | MLflow for logging metrics, parameters, and artifacts |
| 🧱 **CI/CD Pipeline** | Automated build, test, and deploy via GitHub Actions |
| 🐳 **Docker Support** | Preconfigured Dockerfile for containerized deployment |
| 🌐 **Cloud Hosting** | Streamlit Cloud deployment with pre-trained model |
| 📦 **Modular Codebase** | Organized into folders for config, API, data, docs, and logs |
| 💾 **Persistence** | Models and encoders saved with `joblib` |
| 📈 **Business Analytics** | Power BI report with KPIs, churn rate by contract, tenure, and revenue trends |

---

## 🧩 Architecture Diagram

```text
            ┌────────────────────┐
            │   Raw Data (.csv)  │
            └────────┬───────────┘
                     │
        ┌────────────▼─────────────┐
        │  Data Preprocessing      │
        │ (Cleaning, Encoding)     │
        └────────────┬─────────────┘
                     │
            ┌────────▼────────┐
            │ Model Training  │
            │  (XGBoost)      │
            └────────┬────────┘
                     │
        ┌────────────▼────────────┐
        │ Evaluation + MLflow Log │
        └────────────┬────────────┘
                     │
            ┌────────▼─────────┐
            │   Model Artifacts│
            │ (.pkl, scaler)   │
            └────────┬─────────┘
                     │
       ┌─────────────▼──────────────┐
       │ Streamlit App (Frontend UI)│
       └─────────────┬──────────────┘
                     │
            ┌────────▼────────┐
            │ Power BI Report │
            └─────────────────┘
````

---

## 📁 Project Structure

```
📦 End-to-End-ML-Project-Customer-Churn-Prediction
├── api/
│   ├── main.py
│   ├── schemas/
│   └── utils/
├── config/
│   └── model_config.yaml
├── dashboard/
│   ├── powerbi/
│   └── streamlit/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── predictions/
│   └── README.md
├── docs/
│   ├── business_report.pdf
│   ├── model_evaluation.md
│   └── images/
│       ├── shap_plot.png
│       ├── roc_curve.png
│       ├── precision_recall.png
├── models/
│   └── artifacts/
│       ├── best_model.pkl
│       ├── scaler.pkl
│       ├── label_encoders.pkl
│       └── feature_names.pkl
├── mlruns/
│   └── tracking metadata
├── notebooks/
│   ├── EDA.ipynb
│   ├── Feature_Engineering.ipynb
│   └── Model_Training.ipynb
├── streamlit_app.py
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml
└── README.md
```

---

## 🧰 Tech Stack

| Layer                   | Technology                                      |
| ----------------------- | ----------------------------------------------- |
| **Frontend**            | Streamlit (1.36.0)                              |
| **Backend / ML**        | Python 3.10.11, scikit-learn, XGBoost, LightGBM |
| **Experiment Tracking** | MLflow                                          |
| **Containerization**    | Docker                                          |
| **Visualization**       | Power BI, Matplotlib, Seaborn                   |
| **Deployment**          | Streamlit Cloud                                 |
| **Automation**          | GitHub Actions (CI/CD)                          |
| **Storage**             | Joblib model serialization                      |

---

## 🧮 Model Details

* **Algorithms tested:** RandomForest, XGBoost, LightGBM
* **Selected model:** XGBoost (`best_model.pkl`)
* **Metrics:**

  * Accuracy: 0.86
  * Precision: 0.83
  * Recall: 0.78
  * F1-score: 0.80
  * ROC AUC: 0.89

---

## 🧾 Streamlit App Features

🔹 Intuitive sidebar form for customer input
🔹 Real-time churn probability prediction
🔹 Clean gradient background with modern UI
🔹 Dynamic KPIs and insights display
🔹 Custom logo and color palette
🔹 Pretrained model artifacts auto-loaded

---

## 🌐 Live Demo

👉 **Try it here:**
[Customer Churn Prediction App](https://alizarkhan62--end-to-end-ml-project-custom-streamlit-app-0sdl14.streamlit.app/)

---

## 🪄 Screenshots (Placeholders)

| Section                                 | Screenshot                                              |
| --------------------------------------- | ------------------------------------------------------- |
| **Streamlit App Homepage**              | ![Streamlit App UI](docs/images/streamlit_ui.png)       |
| **Prediction Output**                   | ![Prediction Result](docs/images/prediction_result.png) |
| **Power BI Page 1 (Overview)**          | ![Power BI Overview](docs/images/powerbi_page1.png)     |
| **Power BI Page 2 (Customer Analysis)** | ![Power BI Analysis](docs/images/powerbi_page2.png)     |
| **MLflow Tracking**                     | ![MLflow UI](docs/images/mlflow_tracking.png)           |
| **Dockerized Deployment**               | ![Docker Setup](docs/images/docker_setup.png)           |

---

## 🧩 Power BI Dashboard Highlights

### **Page 1 — Executive Overview**

* KPIs: Total Customers, Churn Rate, Avg Monthly Charges, Avg Tenure, High-Risk Count
* Visuals: Contract-wise churn, Internet service churn, CLV distribution, Monthly trends
* Navigation buttons and slicers for gender, contract, payment type

### **Page 2 — Customer Analysis**

* Detailed churn probability by customer demographics
* Revenue at risk, retention ROI, and campaign recommendations
* Drill-through filters, bookmarks, and KPI comparison

---

## ⚙️ Setup Instructions

### 🖥 Local Setup

```bash
# Clone the repository
git clone https://github.com/AlizarKhan62/-End-to-End-ML-Project-Customer-Churn-Predition.git
cd -End-to-End-ML-Project-Customer-Churn-Predition

# Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

---

### 🐳 Docker Setup

```bash
# Build Docker image
docker build -t churn-prediction-app .

# Run container
docker run -p 8501:8501 churn-prediction-app
```

---

### ☁️ Streamlit Cloud Deployment

1. Push all files (including `models/artifacts` and `requirements.txt`) to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io) → “Deploy App”.
3. Set:

   * **Repository:** `AlizarKhan62/-End-to-End-ML-Project-Customer-Churn-Predition`
   * **Branch:** `main`
   * **Main file path:** `streamlit_app.py`
4. Done 🎉 Your app will build and deploy automatically.

---

## 🔗 CI/CD Workflow

GitHub Actions pipeline (`.github/workflows/ci.yml`) performs:

* Code quality check
* Dependency installation
* Model training validation
* Unit tests (pytest)
* Build & deploy to Streamlit Cloud

---

## 🧾 Business Impact

✅ Identify at-risk customers early
✅ Predict churn probability per user
✅ Enable retention campaigns with high ROI
✅ Reduce churn by data-driven segmentation



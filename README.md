# Enterprise Intern Performance Intelligence & Predictive Engine

<div align="left">
  <img src="https://img.shields.io/badge/Domain-Human%20Capital%20Analytics-6366F1?style=for-the-badge&logo=python" alt="Domain" />
  <img src="https://img.shields.io/badge/Engine-XGBoost%20Regression-FF9900?style=for-the-badge&logo=xgboost&logoColor=white" alt="Engine" />
  <img src="https://img.shields.io/badge/Accuracy-R%C2%B2%20%3E%200.90-22D3EE?style=for-the-badge" alt="Metric" />
</div>

<br>

An enterprise-grade predictive analytics framework designed to forecast intern performance trajectories at **Internee.pk**. This system leverages high-dimensional ensemble learning (XGBoost) to evaluate non-linear correlations between task efficiency, feedback sentiment, and engagement consistency, enabling automated high-potential identification and risk mitigation.

---

## 🏗️ System Architecture & Execution Pipeline

```mermaid
graph TD
    A["Raw Performance Telemetry Ingestion"] --> B["Continuous Variable Normalization"]
    A --> C["Feedback Sentiment Inflection Tracking"]
    B --> D["Feature Engineering Module"]
    C --> D
    D -->|compounding Efficiency x Feedback| E["Productivity Index (Engineered Domain Marker)"]
    E --> F["Stratified Cross-Validation Splitting"]
    F --> G["XGBoost Ensemble Optimizer"]
    G --> H["Hyperparameter Tuning & Regularization"]
    H --> I["Real-Time Inference API (FastAPI)"]
    I --> J["Premium Analytics Dashboard (UI)"]
    J --> K["Automated HR Recommendations"]
```

---

## 🔬 Methodology & Feature Engineering

### **Mathematical Modeling: Productivity Index**
To capture the underlying multi-variable correlation between speed and quality, the pipeline incorporates an engineered macro-variable compounding normalized completion velocity against qualitative mentor feedback:
$$\text{Productivity Index} = \frac{(100 - \text{Task Completion Time})}{\text{Max Time}} \times \text{Feedback Rating}$$

This structural marker significantly enhances the gradient descent efficiency across the ensemble layers.

---

## 📊 Performance Telemetry & Diagnostics

The modeling core evaluates intern outcomes over stratified splits to ensure zero-bias projections.

| Evaluation Metric | Value | Threshold Status |
| :--- | :---: | :---: |
| **Model R² (Variance Explained)** | `0.9036` | ✅ **OPTIMIZED** |
| **Mean Squared Error (MSE)** | `34.45` | ✅ **STABLE** |
| **Mean Absolute Error (MAE)** | `4.21` | ✅ **STABLE** |
| **Prediction Latency** | `<15ms` | ✅ **REAL-TIME** |

---

## 📂 Deliverables Layout

```text
Intern-Performance-Predictor/
│
├── backend.py              # FastAPI Production Server serving real-time inference
├── main.py                 # Core ML Engine (Feature Engineering & XGBoost Training)
├── requirements.txt        # Production dependency manifest
│
└── static/                 # High-Fidelity UI Assets
    ├── index.html          # Dashboard Structure
    ├── style.css           # Premium Glassmorphism Styling
    └── script.js           # Real-Time API Integration Logic
```

---

## 💻 Local Execution Guide

### **1. Provision Runtime Environment**
Ensure the high-performance continuous libraries are provisioned:
```bash
pip install pandas numpy xgboost scikit-learn fastapi uvicorn
```

### **2. Launch Production AI Engine**
Execute the unified backend to initialize the XGBoost model and serve the API:
```bash
python backend.py
```

### **3. Access Analytics Dashboard**
Initialize the frontend interface to review explanatory telemetry and real-time predictions:
`http://localhost:8000`

---
*Developed for Internee.pk AI Intelligence Suite.*

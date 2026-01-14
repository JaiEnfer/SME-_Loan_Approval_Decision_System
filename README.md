# SME Loan Approval Decision System

An end-to-end, **finance-grade SME loan approval decision system** that combines  **machine learning**, **business policy rules**, and **explainability (SHAP)**,  
deployed as an interactive **Streamlit application**.

This project mirrors how **real fintechs and banks** build auditable, risk-aware credit decision engines.

---

## Project Overview

Small and Medium Enterprises (SMEs) are a critical part of the economy, but lending to SMEs involves:
- high class imbalance (few good vs many risky applicants),
- strict risk management requirements,
- regulatory and explainability constraints.

This project addresses those challenges by implementing a **hybrid ML + rule-based decision system**.

---

## Key Features

### Machine Learning–Based Risk Scoring
- RandomForest classifier trained on SME credit, legal, tax, and operational features
- Outputs **approval probability** instead of just a binary decision
- Handles severe class imbalance using balanced learning strategies

### Business Policy Overrides (Bank-Style)
- Rule-based overrides layered on top of ML predictions
- Examples:
  - Recent overdue tax → auto reject
  - Recent legal proceedings → auto reject
  - Very new businesses → flagged for review
- Generates **audit reasons** for every overridden decision

### Risk-Based Decisions
Instead of a simple approve/reject output, the system produces:
- **Approve**
- **Review**
- **Reject**

This reflects real-world SME credit workflows.

### Explainability with SHAP
- SHAP waterfall plots explain **why a specific SME was approved or rejected**
- Feature-level transparency suitable for:
  - credit committees
  - compliance reviews
  - client communication

### Interactive Streamlit Application
- Upload SME data via **CSV or Excel**
- Adjustable decision threshold (risk appetite control)
- Downloadable decision results with audit logs
- Visual SHAP explanations per SME

---

### Data
[Data can be downloaded from Kaggle also] {https://www.kaggle.com/datasets/cr30srtfcn/msme-credit-data-by-30scr?utm_source=chatgpt.com}

---


## Decision Architecture
```text
SME Input Data
↓
Data Cleaning & Feature Alignment
↓
ML Model (Approval Probability)
↓
Threshold-Based Decision
↓
Policy Override Engine
↓
Final Decision + Audit Reason
```

---

## Project Structure

```text
SME Loan Approval Decision System/
├── app/
│ └── app.py # Streamlit application
├── models/
│ ├── rf_loan_model.joblib # Trained RandomForest model
│ ├── train_columns.joblib # Feature schema
│ └── best_threshold.joblib # Optimized decision threshold
├── notebooks/
│ └── exploration_and_model.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```


---

## ⚙️ Tech Stack

- **Python**
- **Pandas / NumPy** – data processing
- **Scikit-learn** – machine learning
- **SHAP** – explainability
- **Streamlit** – application & UI
- **Joblib** – model persistence
- **Matplotlib** – visualizations

---

## How to Run Locally

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the Streamlit app
```bash
streamlit run app/app.py
```

### Input Data

- Accepts CSV or Excel (.xlsx) files
- Automatically:
    -  handles encoding and separators,
    -  imputes missing values,
    -  aligns features to the training schema.  

### Explainability Example (SHAP)

For any selected SME, the app shows:

- top positive factors contributing to approval,
- top negative risk drivers,
- base risk vs final score.

This enables transparent and defensible credit decisions.
---
## Notes

Model artifacts are stored using joblib.

If models are excluded from the repo, they can be regenerated via the notebook.

The decision threshold can be adjusted to reflect different risk appetites.

---

## UI Screenshots

<img width="1915" height="872" alt="1" src="https://github.com/user-attachments/assets/41ffa43b-fc23-4253-95a9-7509d7b017e0" />
<img width="1906" height="863" alt="2" src="https://github.com/user-attachments/assets/0cd6e179-f2ca-4f65-a2e9-d13af185c23d" />
<img width="1917" height="876" alt="3" src="https://github.com/user-attachments/assets/917095d0-4c6c-45e4-a8d7-a9934dfeaaa5" />

---

___Thank You___

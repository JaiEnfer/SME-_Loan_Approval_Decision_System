# app.py
# Streamlit SME Loan Approval Decision System (CSV/XLSX upload + ML + Policy Overrides + SHAP)
# -----------------------------------------------------------------------------
# Recommended structure:
# SME Loan Approval Decision System/
#   app/app.py
#   models/
#     rf_loan_model.joblib
#     train_columns.joblib
#     best_threshold.joblib
#
# Install deps in the SAME environment you run streamlit:
#   python -m pip install -U streamlit pandas numpy scikit-learn joblib shap matplotlib openpyxl

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -----------------------------------------------------------------------------
# Paths (robust across OS)
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent          # .../app
MODEL_DIR = (BASE_DIR / ".." / "models").resolve()  # .../models

# -----------------------------------------------------------------------------
# Streamlit config
# -----------------------------------------------------------------------------
st.set_page_config(page_title="SME Loan Approval Decision System", layout="wide")
st.title("🏦 SME Loan Approval Decision System")
st.write("Upload SME data (**CSV or Excel**) → get approval probability, decisions, risk actions, and audit reasons.")

# -----------------------------------------------------------------------------
# Load model artifacts
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = MODEL_DIR / "rf_loan_model.joblib"
    cols_path  = MODEL_DIR / "train_columns.joblib"
    thr_path   = MODEL_DIR / "best_threshold.joblib"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not cols_path.exists():
        raise FileNotFoundError(f"Missing columns file: {cols_path}")
    if not thr_path.exists():
        raise FileNotFoundError(f"Missing threshold file: {thr_path}")

    model = joblib.load(model_path)
    train_cols = joblib.load(cols_path)
    threshold = float(joblib.load(thr_path))
    return model, train_cols, threshold


# -----------------------------------------------------------------------------
# File loading (CSV + Excel) with encoding + separator fallbacks
# -----------------------------------------------------------------------------
@st.cache_data
def load_input_file(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name
    ext = os.path.splitext(name)[1].lower()

    if ext == ".xlsx":
        return pd.read_excel(uploaded_file)  # needs openpyxl

    if ext == ".csv":
        def _try_read(encoding: str, sep: str):
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding, sep=sep)

        attempts = [
            ("utf-8", ","),
            ("utf-8-sig", ","),
            ("latin1", ","),
            ("utf-8", ";"),
            ("utf-8-sig", ";"),
            ("latin1", ";"),
        ]
        last_err = None
        for enc, sep in attempts:
            try:
                return _try_read(enc, sep)
            except Exception as e:
                last_err = e
        raise last_err

    raise ValueError("Unsupported file type. Please upload a .csv or .xlsx file.")


# -----------------------------------------------------------------------------
# Basic imputations (safe defaults for client uploads)
# -----------------------------------------------------------------------------
def basic_impute(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Coerce numerics where possible
    num_cols = out.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    if len(num_cols) > 0:
        out[num_cols] = out[num_cols].apply(pd.to_numeric, errors="coerce")
        out[num_cols] = out[num_cols].fillna(out[num_cols].median())

    # Handle categoricals
    cat_cols = out.select_dtypes(include=["object", "category"]).columns
    for c in cat_cols:
        if out[c].isna().all():
            out[c] = out[c].fillna("UNKNOWN")
        else:
            out[c] = out[c].fillna(out[c].mode().iloc[0])

    return out


# -----------------------------------------------------------------------------
# Align incoming columns to training features (one-hot + add missing + drop extra)
# -----------------------------------------------------------------------------
def align_to_training_columns(df: pd.DataFrame, train_cols: list[str]) -> pd.DataFrame:
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Add missing
    for c in train_cols:
        if c not in df_enc.columns:
            df_enc[c] = 0

    # Drop extras & reorder
    df_enc = df_enc[train_cols]

    # Ensure numeric matrix
    df_enc = df_enc.apply(pd.to_numeric, errors="coerce").fillna(0)
    return df_enc


# -----------------------------------------------------------------------------
# Policy overrides (bank-style) + audit reasons
# -----------------------------------------------------------------------------
def apply_policy_overrides(raw_df: pd.DataFrame):
    X = raw_df
    n = len(X)
    override_flag = np.zeros(n, dtype=bool)
    override_decision = np.full(n, -1, dtype=int)  # -1 => none
    reasons = [[] for _ in range(n)]

    def _flag(mask, decision, reason):
        idxs = np.where(mask)[0]
        for i in idxs:
            override_flag[i] = True
            override_decision[i] = decision
            reasons[i].append(reason)

    # Hard reject rules
    if "Overdue_tax_num_1year" in X.columns:
        mask = pd.to_numeric(X["Overdue_tax_num_1year"], errors="coerce").fillna(0) > 0
        _flag(mask, 0, "Policy: Overdue tax in last 1 year")

    if "Executee_num_1year" in X.columns:
        mask = pd.to_numeric(X["Executee_num_1year"], errors="coerce").fillna(0) > 0
        _flag(mask, 0, "Policy: Executee record in last 1 year")

    if "Legal_proceedings_num_1year" in X.columns:
        mask = pd.to_numeric(X["Legal_proceedings_num_1year"], errors="coerce").fillna(0) >= 1
        _flag(mask, 0, "Policy: Legal proceedings in last 1 year")

    # Soft flag (review hint)
    if "Establishment_Duration (Days)" in X.columns:
        mask = pd.to_numeric(X["Establishment_Duration (Days)"], errors="coerce").fillna(0) < 365
        idxs = np.where(mask)[0]
        for i in idxs:
            reasons[i].append("Flag: New business (<1 year)")

    reasons = ["; ".join(r) if r else "" for r in reasons]
    return override_flag, override_decision, reasons


# -----------------------------------------------------------------------------
# Decision engine (ML threshold + overrides + 3-action output)
# -----------------------------------------------------------------------------
def decision_engine(raw_df: pd.DataFrame, X_aligned: pd.DataFrame, model, threshold: float) -> pd.DataFrame:
    proba = model.predict_proba(X_aligned)[:, 1]
    ml_decision = (proba >= threshold).astype(int)

    override_flag, override_decision, reasons = apply_policy_overrides(raw_df)

    final_decision = ml_decision.copy()
    final_decision[override_flag] = override_decision[override_flag]

    # 3 actions (business-friendly)
    action = pd.cut(
        proba,
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Reject", "Review", "Approve"],
        include_lowest=True
    )

    out = raw_df.copy()
    out["approval_prob"] = proba
    out["ml_decision"] = ml_decision
    out["policy_overridden"] = override_flag
    out["final_decision"] = final_decision
    out["action"] = action.astype(str)
    out["audit_reason"] = reasons
    return out


# -----------------------------------------------------------------------------
# SHAP waterfall (robust for RandomForest across SHAP versions)
# -----------------------------------------------------------------------------
def make_shap_waterfall(model, X_row: pd.DataFrame):
    """
    Robust SHAP waterfall for RandomForest that returns a matplotlib Figure
    (compatible with Streamlit st.pyplot).
    """
    import shap
    import numpy as np
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model)
    shap_row = explainer.shap_values(X_row)

    # Pick class 1 for binary classification
    if isinstance(shap_row, list):
        sv = np.array(shap_row[1])
        base = explainer.expected_value[1]
    else:
        sv = np.array(shap_row)
        base = explainer.expected_value
        if sv.ndim == 3:
            sv = sv[:, :, 1]
            base = base[1]

    # Remove bias column if present
    if sv.shape[1] == X_row.shape[1] + 1:
        sv = sv[:, :-1]

    exp = shap.Explanation(
        values=sv[0],
        base_values=base,
        data=X_row.iloc[0],
        feature_names=list(X_row.columns),
    )

    # --- KEY FIX ---
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.plots.waterfall(exp, max_display=15, show=False)
    fig = plt.gcf()  # get the current Figure (not Axes)
    plt.tight_layout()

    return fig


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(f"Artifacts folder: {MODEL_DIR}")
    show_debug = st.checkbox("Show debug info", value=False)

model, train_cols, default_threshold = load_artifacts()

threshold = st.sidebar.slider(
    "Decision threshold (Approve if prob ≥ threshold)",
    0.01, 0.99, float(default_threshold), 0.01
)

show_shap = st.sidebar.checkbox("Show SHAP explanation (selected row)", value=True)


# -----------------------------------------------------------------------------
# Upload + load
# -----------------------------------------------------------------------------
uploaded = st.file_uploader("Upload SME data (CSV or Excel)", type=["csv", "xlsx"])

if uploaded is None:
    st.info("Upload a file to get predictions.")
    st.stop()

raw = load_input_file(uploaded)

if show_debug:
    st.write("Loaded file:", uploaded.name)
    st.write("Raw shape:", raw.shape)
    st.dataframe(raw.head(5), use_container_width=True)

# Drop ID if present (not predictive)
if "Enterprise_id" in raw.columns:
    raw = raw.drop(columns=["Enterprise_id"])

raw_clean = basic_impute(raw)

# Align to training columns
X_in = align_to_training_columns(raw_clean, train_cols)

# -----------------------------------------------------------------------------
# Score
# -----------------------------------------------------------------------------
result = decision_engine(raw_clean, X_in, model, threshold)

st.subheader("📊 Decisions")
st.write(f"Rows scored: {len(result)}  |  Model features: {X_in.shape[1]}")
st.dataframe(result.head(50), use_container_width=True)

st.download_button(
    "⬇️ Download decisions CSV",
    data=result.to_csv(index=False).encode("utf-8"),
    file_name="loan_decisions.csv",
    mime="text/csv"
)

with st.expander("🔍 Preview uploaded data (first 20 rows)"):
    st.dataframe(raw_clean.head(20), use_container_width=True)


# -----------------------------------------------------------------------------
# SHAP (explain one row)
# -----------------------------------------------------------------------------
if show_shap:
    st.subheader("🧠 SHAP Explanation (one SME)")
    idx = st.number_input("Row index to explain", min_value=0, max_value=len(X_in) - 1, value=0, step=1)

    try:
        # Explain a single aligned row
        x_row = X_in.iloc[[int(idx)]]
        fig = make_shap_waterfall(model, x_row)
        st.pyplot(fig, clear_figure=True)
    except Exception as e:
        st.error("SHAP failed with this error (install/upgrade shap in this environment):")
        st.exception(e)

import streamlit as st
import pandas as pd, numpy as np, joblib
import altair as alt
from pathlib import Path

st.set_page_config(page_title="Claims Optimization System", layout="wide")

DATA_PATH = Path("data/claims.csv")
MODEL_PATH = Path("models/claim_approval.joblib")

@st.cache_data
def load_data(path: Path):
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

@st.cache_resource
def load_model(path: Path):
    return joblib.load(path) if path.exists() else None

df = load_data(DATA_PATH)
bundle = load_model(MODEL_PATH)
model = bundle["model"] if bundle else None
meta = bundle["meta"] if bundle else {}

# -------------------
# Normalization Helper
# -------------------
REQUIRED_COLUMNS = [
    "claim_amount","patient_age","patient_gender","insurance_provider",
    "diagnosis_code","procedure_code","in_network","preauth_obtained",
    "prior_denials_count","days_since_service"
]

COLUMN_ALIASES = {
    "age": "patient_age",
    "gender": "patient_gender",
    "provider": "insurance_provider",
    "claim": "claim_amount",
    "diag_code": "diagnosis_code",
    "proc_code": "procedure_code",
    "network": "in_network",
    "preauth": "preauth_obtained",
    "denials": "prior_denials_count",
    "days": "days_since_service"
}

DEFAULTS = {
    "claim_amount": 0.0,
    "patient_age": 40,
    "patient_gender": "Unknown",
    "insurance_provider": "Unknown",
    "diagnosis_code": "UNK",
    "procedure_code": "UNK",
    "in_network": 0,
    "preauth_obtained": 0,
    "prior_denials_count": 0,
    "days_since_service": 0
}

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Rename known aliases
    df = df.rename(columns=COLUMN_ALIASES)

    # Add missing required columns with defaults
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = DEFAULTS[col]

    # Ensure correct column order
    return df[REQUIRED_COLUMNS]

# -------------------
# UI Starts
# -------------------
st.title("Claims Optimization System")
st.caption("Healthcare claim approval prediction • Fraud & denial insights")

st.sidebar.header("Settings")
uploaded = st.sidebar.file_uploader("Upload claims CSV (optional)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)

threshold = st.sidebar.slider("Approval threshold (P ≥)", 0.05, 0.95, float(meta.get("threshold_default", 0.5)), 0.01)
show_raw = st.sidebar.checkbox("Show raw data sample", value=False)

tab_dash, tab_predict = st.tabs(["📊 Dashboards", "🧮 Predictor"])

# -------------------
# Dashboard Tab
# -------------------
with tab_dash:
    st.subheader("Portfolio Overview")
    if df.empty:
        st.info("No data found. Upload a CSV in the sidebar (or generate one with the sample script).")
    else:
        if "status" in df.columns:
            status_counts = df["status"].value_counts(dropna=False).rename_axis("status").reset_index(name="count")
            st.metric("Total claims", int(len(df)))
            c1, c2, c3 = st.columns(3)
            c1.metric("Approved", int((df["status"].astype(str).str.lower()=="approved").sum()))
            c2.metric("Denied", int((df["status"].astype(str).str.lower()=="denied").sum()))
            if "claim_amount" in df.columns:
                c3.metric("Avg claim amount", f"${df['claim_amount'].mean():,.0f}")
            chart = alt.Chart(status_counts).mark_bar().encode(
                x=alt.X("status:N", title="Claim Status"),
                y=alt.Y("count:Q", title="Count"),
                tooltip=["status","count"]
            ).properties(height=280)
            st.altair_chart(chart, use_container_width=True)

        if "claim_amount" in df.columns:
            dist = alt.Chart(df).transform_bin("bin_amount", field="claim_amount", bin=alt.Bin(maxbins=40)) \
                                .mark_bar().encode(
                x=alt.X("bin_amount:Q", title="Claim amount (binned)"),
                y=alt.Y("count()", title="Count"),
                tooltip=[alt.Tooltip("count()", title="Count")]
            ).properties(height=280)
            st.subheader("Claim Amount Distribution")
            st.altair_chart(dist, use_container_width=True)

        if {"insurance_provider","claim_amount"}.issubset(df.columns):
            prov = (df.groupby("insurance_provider", as_index=False)
                      .agg(avg_amount=("claim_amount","mean"), count=("claim_amount","count")))
            st.subheader("Insurance Provider Trends")
            colA, colB = st.columns([2,1])
            chartA = alt.Chart(prov).mark_bar().encode(
                x=alt.X("insurance_provider:N", sort="-y", title="Provider"),
                y=alt.Y("count:Q", title="Claims"),
                tooltip=["insurance_provider","count","avg_amount"]
            ).properties(height=280)
            colA.altair_chart(chartA, use_container_width=True)
            chartB = alt.Chart(prov).mark_circle(size=120).encode(
                x=alt.X("insurance_provider:N", sort="-y", title="Provider"),
                y=alt.Y("avg_amount:Q", title="Avg $"),
                tooltip=["insurance_provider","avg_amount","count"]
            ).properties(height=280)
            colB.altair_chart(chartB, use_container_width=True)

        if {"denial_reason","status"}.issubset(df.columns):
            dr = (df.assign(denial_reason=df["denial_reason"].replace("", np.nan))).dropna(subset=["denial_reason"])
            if not dr.empty:
                st.subheader("Top Denial Reasons")
                topd = (dr[dr["status"].astype(str).str.lower()=="denied"]
                        .groupby("denial_reason", as_index=False).size()
                        .rename(columns={"size":"count"})
                        .sort_values("count", ascending=False).head(15))
                chartDR = alt.Chart(topd).mark_bar().encode(
                    x=alt.X("count:Q", title="Count"),
                    y=alt.Y("denial_reason:N", sort="-x", title="Reason"),
                    tooltip=["denial_reason","count"]
                ).properties(height=400)
                st.altair_chart(chartDR, use_container_width=True)

        if show_raw:
            st.subheader("Raw data sample")
            st.dataframe(df.head(50), use_container_width=True)

# -------------------
# Predictor Tab
# -------------------
with tab_predict:
    st.subheader("Single Claim Scoring")
    if not model:
        st.warning("No trained model found. Run `python train.py` first.")
    else:
        with st.form("single_claim"):
            c1, c2, c3 = st.columns(3)
            amount = c1.number_input("Claim amount", min_value=0.0, value=1200.0, step=50.0)
            age = c2.number_input("Patient age", min_value=0, max_value=120, value=44)
            gender = c3.selectbox("Patient gender", ["F","M","O"])
            provider = c1.text_input("Insurance provider", "Aetna")
            diag = c2.text_input("Diagnosis code", "I042")
            proc = c3.text_input("Procedure code", "P123")
            in_net = c1.selectbox("In network?", ["No","Yes"])
            preauth = c2.selectbox("Pre-auth obtained?", ["No","Yes"])
            prior_denials = c3.number_input("Prior denials count", min_value=0, value=0)
            days_since = c1.number_input("Days since service", min_value=0, value=14)
            submitted = st.form_submit_button("Predict")

        if submitted:
            row = pd.DataFrame([{
                "claim_id": "NEW",
                "claim_amount": amount,
                "patient_age": age,
                "patient_gender": gender,
                "insurance_provider": provider,
                "diagnosis_code": diag,
                "procedure_code": proc,
                "in_network": 1 if in_net=="Yes" else 0,
                "preauth_obtained": 1 if preauth=="Yes" else 0,
                "prior_denials_count": prior_denials,
                "days_since_service": days_since
            }])

            row = normalize_df(row)

            p_approved = float(model.predict_proba(row)[0,1])
            approved = p_approved >= threshold
            st.metric("P(Approved)", f"{p_approved:.2%}")
            st.success("Prediction: APPROVED ✅") if approved else st.error("Prediction: DENIED ❌")
            st.caption(f"Threshold = {threshold:.2f} • Adjust in sidebar.")

        st.divider()
        st.subheader("Batch Scoring (CSV)")
        batch_file = st.file_uploader("Upload CSV to score (any format – auto-normalized)", type=["csv"], key="batch")
        if batch_file:
            bdf = pd.read_csv(batch_file)
            bdf = normalize_df(bdf)

            proba = model.predict_proba(bdf)[:,1]
            preds = (proba >= threshold).astype(int)

            out = bdf.copy()
            out["p_approved"] = proba
            out["predicted_status"] = np.where(preds==1,"Approved","Denied")

            st.write("Preview:")
            st.dataframe(out.head(20), use_container_width=True)
            st.download_button(
                "⬇️ Download scored CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="scored_claims.csv",
                mime="text/csv"
            )

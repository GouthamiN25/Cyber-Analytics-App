# streamlit_cyber_app.py
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from matplotlib.ticker import MaxNLocator
from scipy.sparse import hstack
import joblib

st.set_page_config(page_title="SOC Dashboard", layout="wide")
st.title("🔐 Cybersecurity Threat Analysis & RAG Playbook")

# ------------------ DATA SOURCE ------------------
# Edit this to your exact CSV path if you want a hardcoded default:
HARDCODED = "/Users/gouthami/Downloads/cyber-data/merged_cyber_incidents.csv"
CWD_CANDIDATE = os.path.join(os.getcwd(), "merged_cyber_incidents.csv")

st.sidebar.markdown("### Data Source")
use_upload = st.sidebar.checkbox("Upload CSV instead")

MERGED = None
if use_upload:
    uploaded = st.sidebar.file_uploader("Upload merged_cyber_incidents.csv", type=["csv"])
    if uploaded is not None:
        MERGED = uploaded  # file-like object
else:
    if os.path.exists(HARDCODED):
        MERGED = HARDCODED
    elif os.path.exists(CWD_CANDIDATE):
        MERGED = CWD_CANDIDATE

if MERGED is None:
    st.error("CSV not found. Put merged_cyber_incidents.csv here, edit HARDCODED, or upload via sidebar.")
    st.stop()

df = pd.read_csv(MERGED)
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

# ------------------ FILTERS ------------------
sev_opts  = ["all"] + (sorted(df["severity"].dropna().unique().tolist()) if "severity" in df.columns else [])
type_opts = ["all"] + (sorted(df["threat_type"].dropna().unique().tolist()) if "threat_type" in df.columns else [])
dept_opts = ["all"] + (sorted(df["asset_owner_department"].dropna().unique().tolist()) if "asset_owner_department" in df.columns else [])

f_sev  = st.sidebar.selectbox("Filter severity", sev_opts)
f_type = st.sidebar.selectbox("Filter threat type", type_opts)
f_dept = st.sidebar.selectbox("Filter department", dept_opts)

mask = pd.Series(True, index=df.index)
if f_sev != "all" and "severity" in df.columns:
    mask &= (df["severity"] == f_sev)
if f_type != "all" and "threat_type" in df.columns:
    mask &= (df["threat_type"] == f_type)
if f_dept != "all" and "asset_owner_department" in df.columns:
    mask &= (df["asset_owner_department"] == f_dept)

dff = df.loc[mask].copy()

st.subheader("Dataset Preview")
st.dataframe(dff.head(30))

# ------------------ KPIs ------------------
col1,col2,col3,col4 = st.columns(4)
col1.metric("Incidents (filtered)", len(dff))
col2.metric("Unique assets", (dff["asset_id"].nunique() if "asset_id" in dff.columns else 0))
col3.metric("High/Critical", int(dff["severity"].isin(["high","critical"]).sum()) if "severity" in dff.columns else 0)
col4.metric("Open/Investigating", int(dff["status"].isin(["open","investigating"]).sum()) if "status" in dff.columns else 0)

# ------------------ CHARTS ------------------
if "timestamp" in dff.columns and pd.api.types.is_datetime64_any_dtype(dff["timestamp"]):
    daily = dff.set_index("timestamp").resample("D").size()
    if len(daily):
        fig = plt.figure()
        plt.plot(daily.index, daily.values)
        plt.title("Daily Incident Volume")
        plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        st.pyplot(fig)

for col, title in [("threat_type","Threat Type Distribution"), ("severity","Severity Distribution"), ("status","Status Distribution")]:
    if col in dff.columns:
        counts = dff[col].value_counts()
        fig = plt.figure()
        counts.plot(kind="bar")
        plt.title(title)
        plt.tight_layout()
        st.pyplot(fig)

# ------------------ ARTIFACTS (optional models) ------------------
ART_DIR = os.getcwd()
def load_artifact(name):
    p = os.path.join(ART_DIR, name)
    return joblib.load(p) if os.path.exists(p) else None

tfidf  = load_artifact("tfidf_vectorizer.joblib")
ohe    = load_artifact("onehot_encoder.joblib")
scaler = load_artifact("numeric_scaler.joblib")
clf_sev= load_artifact("clf_severity_logreg.joblib")
clf_hc = load_artifact("clf_highcritical_logreg.joblib")

# ------------------ PREDICTOR ------------------
st.subheader("🔮 Predict Severity (baseline)")
txt = st.text_area("Incident description", "File encryption detected on shared drive; ransom note present.")
tt  = st.selectbox("Threat type", ["unknown"] + (sorted(dff["threat_type"].fillna("unknown").unique().tolist()) if "threat_type" in dff.columns else []))
stt = st.selectbox("Status", ["unknown"] + (sorted(dff["status"].fillna("unknown").unique().tolist()) if "status" in dff.columns else []))
at  = st.selectbox("Asset type", ["unknown"] + (sorted(dff["asset_type"].fillna("unknown").unique().tolist()) if "asset_type" in dff.columns else []))
dept= st.selectbox("Owner dept", ["unknown"] + (sorted(dff["asset_owner_department"].fillna("unknown").unique().tolist()) if "asset_owner_department" in dff.columns else []))
dow = st.selectbox("Day of week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
hour= st.slider("Hour", 0, 23, 10)
month = st.slider("Month", 1, 12, 6)

if st.button("Predict"):
    if not all([tfidf, ohe, scaler, clf_sev]):
        st.warning("Model artifacts not found in this folder (tfidf_vectorizer.joblib, onehot_encoder.joblib, numeric_scaler.joblib, clf_severity_logreg.joblib).")
    else:
        Xt = tfidf.transform([txt])
        Xc = ohe.transform([[tt or "unknown", stt or "unknown", at or "unknown", dept or "unknown", dow]])
        Xn = scaler.transform([[0.0,0.0,0.0,hour,month]])
        Xq = hstack([Xt, Xc, Xn])
        sev_pred = clf_sev.predict(Xq)[0]
        st.success(f"Predicted severity: {sev_pred}")
        if clf_hc:
            hc_pred = int(clf_hc.predict(Xq)[0])
            st.info(f"High/Critical flag: {'Yes' if hc_pred==1 else 'No'}")

# ------------------ RAG (TF-IDF fallback) ------------------
st.subheader("🧠 RAG: Retrieve Similar Incidents")
query = st.text_input("Query", "credential phishing via HR portal")
topk = st.slider("Top K", 1, 20, 8)

if st.button("Retrieve"):
    if not tfidf:
        st.warning("TF-IDF not available (no tfidf_vectorizer.joblib in this folder). Train & save artifacts, or integrate embeddings RAG.")
    else:
        # Build TF-IDF matrix over full corpus
        corpus = df["description"].fillna("").astype(str) if "description" in df.columns else pd.Series([""])
        D = tfidf.transform(corpus)
        q = tfidf.transform([query])
        sims = (D @ q.T).toarray().ravel()
        idx = np.argsort(sims)[::-1][:topk]
        cols = ["timestamp","threat_type","severity","status","asset_name","emp_name","description","time_to_resolve_hours"]
        cols = [c for c in cols if c in df.columns]
        hits = df.iloc[idx][cols].reset_index(drop=True)
        st.dataframe(hits)

        # Simple playbook suggestion (FIXED: check columns explicitly)
        if "threat_type" in hits.columns:
            ttypes = hits["threat_type"].value_counts().index.tolist()
        else:
            ttypes = []

        recs = []
        if any("phish" in str(t).lower() for t in ttypes):
            recs += ["Quarantine/purge emails", "Reset creds + enforce MFA", "Block sender/domain; IOC hunt"]
        if any(t in ["malware","ransomware"] for t in ttypes):
            recs += ["Isolate endpoints", "Block hashes/domains; IOC sweep", "Restore from backups; check persistence"]
        if any("ddos" in str(t).lower() for t in ttypes):
            recs += ["Rate-limit/geo-filter; WAF rules", "Engage ISP; scale services"]
        if not recs:
            recs = ["Open incident, triage logs, escalate with SLA"]

        st.write({"recommended_actions": recs})

st.caption("Place merged_cyber_incidents.csv and (optional) *.joblib artifacts in this folder.")

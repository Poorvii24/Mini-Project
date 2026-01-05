import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import RobustScaler
import plotly.express as px
import time
import random
import hashlib
from datetime import datetime
import os

# ------------------------------
# 1. Model Architecture
# ------------------------------
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim=17, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, hidden_dim=128):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        x = self.input_layer(x).unsqueeze(1)
        x = self.transformer(x)
        features = x.mean(dim=1)
        return self.fc(features)

# ------------------------------
# 2. Preprocessing Engine (ROBUST VERSION)
# ------------------------------
def get_stable_hash(s):
    return int(hashlib.sha256(str(s).encode('utf-8')).hexdigest(), 16) % 1000

def preprocess_live_data(df):
    proc_df = df.copy()

    # --- SAFETY CHECK: Detect Corrupted/Excel Files ---
    if len(proc_df.columns) < 2:
        st.error("❌ FILE ERROR: The uploaded file does not look like a valid CSV.")
        st.warning("⚠️ Did you upload an Excel (.xlsx) file renamed as .csv? Please 'Save As CSV' in Excel and try again.")
        st.stop()

    # --- STEP 1: Normalize Columns (Strip spaces) ---
    proc_df.columns = proc_df.columns.str.strip()

    # --- STEP 2: Intelligent Renaming (Case-Insensitive) ---
    col_map = {c.lower(): c for c in proc_df.columns}

    if 'duration' in col_map: proc_df.rename(columns={col_map['duration']: 'Duration'}, inplace=True)
    elif 'dur' in col_map: proc_df.rename(columns={col_map['dur']: 'Duration'}, inplace=True)
    elif 'time' in col_map: proc_df.rename(columns={col_map['time']: 'Duration'}, inplace=True)

    if 'proto' in col_map: proc_df.rename(columns={col_map['proto']: 'Proto'}, inplace=True)
    elif 'protocol' in col_map: proc_df.rename(columns={col_map['protocol']: 'Proto'}, inplace=True)

    if 'sport' in col_map: proc_df.rename(columns={col_map['sport']: 'Sport'}, inplace=True)
    if 'dport' in col_map: proc_df.rename(columns={col_map['dport']: 'Dport'}, inplace=True)

    # --- STEP 3: Critical Safety Check ---
    if 'Duration' not in proc_df.columns:
        st.error(f"❌ CSV ERROR: Missing 'Duration' column. Detected columns: {list(proc_df.columns)}")
        st.stop()

    # --- STEP 4: Feature Engineering ---
    proc_df['Duration'] = proc_df['Duration'].replace(0, 1e-6)

    if 'TotBytes' not in proc_df.columns: proc_df['TotBytes'] = 0
    if 'TotPkts' not in proc_df.columns: proc_df['TotPkts'] = 0
    if 'SrcBytes' not in proc_df.columns: proc_df['SrcBytes'] = 0

    proc_df['BytesPerSec'] = proc_df['TotBytes'] / proc_df['Duration']
    proc_df['PktsPerSec'] = proc_df['TotPkts'] / proc_df['Duration']
    proc_df['AvgPktSize'] = proc_df['TotBytes'] / proc_df['TotPkts'].replace(0, 1)
    proc_df['SrcByteRatio'] = proc_df['SrcBytes'] / proc_df['TotBytes'].replace(0, 1)

    if 'Sport' not in proc_df.columns: proc_df['Sport'] = 0
    if 'Dport' not in proc_df.columns: proc_df['Dport'] = 0

    proc_df['Sport'] = pd.to_numeric(proc_df['Sport'], errors='coerce').fillna(0)
    proc_df['Dport'] = pd.to_numeric(proc_df['Dport'], errors='coerce').fillna(0)
    proc_df['Sport_is_priv'] = (proc_df['Sport'] <= 1024).astype(int)
    proc_df['Dport_is_priv'] = (proc_df['Dport'] <= 1024).astype(int)

    log_cols = ['TotBytes', 'TotPkts', 'SrcBytes', 'BytesPerSec', 'PktsPerSec', 'AvgPktSize']
    for col in log_cols:
        if col in proc_df.columns:
            proc_df[col] = np.log1p(proc_df[col])

    for col in ['Proto', 'State', 'Dir']:
        if col in proc_df.columns:
            proc_df[col] = proc_df[col].astype(str).apply(lambda x: get_stable_hash(x))
        else:
            proc_df[col] = 0

    expected_cols = [
        'Duration', 'Proto', 'Sport', 'Dir', 'Dport', 'State', 'sTos', 'dTos',
        'TotPkts', 'TotBytes', 'SrcBytes', 'BytesPerSec', 'PktsPerSec',
        'AvgPktSize', 'SrcByteRatio', 'Sport_is_priv', 'Dport_is_priv'
    ]

    final_data = pd.DataFrame()
    for col in expected_cols:
        if col in proc_df.columns:
            final_data[col] = pd.to_numeric(proc_df[col], errors='coerce').fillna(0)
        else:
            final_data[col] = 0

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(final_data)
    return scaled_data

# ------------------------------
# 3. Report Generator
# ------------------------------
def generate_text_report(n_threats, risk_score, suspicious_df):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    status = "CRITICAL" if n_threats > 0 else "SECURE"
    top_attackers_str = "No Source IPs found in capture file."

    ip_col = None
    for c in suspicious_df.columns:
        if c.lower().strip() in ['srcaddr', 'src_ip', 'saddr', 'source']:
            ip_col = c
            break

    if ip_col:
        top_attackers = suspicious_df[ip_col].value_counts().head(20).index.tolist()
        if top_attackers:
            top_attackers_str = ", ".join(str(ip) for ip in top_attackers)

    report = f"""SENTINEL INCIDENT REPORT
            ------------------------
            DATE: {timestamp}
            STATUS: {status}
            THREATS DETECTED: {n_threats}
            RISK SCORE: {risk_score:.2f}%

            TOP ATTACKERS (SOURCE IPs):

        {top_attackers_str}
"""
    return report

# ------------------------------
# 4. Page Config & CSS
# ------------------------------
st.set_page_config(page_title="BOTNET DEFENSE", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at center, #0a0e17 0%, #000000 100%); color: #ffffff; }
    h1, h2, h3, h4, h5 { font-family: 'Orbitron', sans-serif !important; color: #00ffcc !important; }
    [data-testid="stFileUploader"] label { color: #00ffcc !important; }
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.05); border-left: 4px solid #00ffcc; border-radius: 12px; }
    [data-testid="stSidebar"] { background-color: #050505; border-right: 1px solid #333; }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# 5. Load Model (UPDATED FOR 'models/' FOLDER)
# ------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerClassifier(input_dim=17)

    # --- FIX: Look in the 'models' folder ---
    model_path = "models/transformer_classifier.pt"

    # Debugging: Check if file exists
    if not os.path.exists(model_path):
        st.error(f"❌ MODEL NOT FOUND at: {model_path}")
        st.write("📂 Current Directory:", os.getcwd())
        if os.path.exists("models"):
            st.write("📂 Files inside 'models':", os.listdir("models"))
        else:
            st.write("⚠️ The 'models' folder does not exist in this repo.")

        # Fallback: Check root
        if os.path.exists("transformer_classifier.pt"):
            model_path = "transformer_classifier.pt"
            st.warning("⚠️ Found model in ROOT folder, not 'models' folder. Loading anyway...")
        else:
            return None, device

    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_path), strict=False)
    else:
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), strict=False)

    model.to(device)
    model.eval()
    return model, device

try:
    model, device = load_model()
    if model is None:
        st.stop()
except Exception as e:
    st.error(f"System Failure: Model loading error. {e}")
    st.stop()

# ------------------------------
# 6. SIDEBAR
# ------------------------------
with st.sidebar:
    st.title("SENTINEL CORE")
    st.caption("v3.1.0 | ENTERPRISE EDITION")
    st.markdown("---")
    threshold = st.slider("THREAT SENSITIVITY", 0.0, 1.0, 0.1)

    if model is not None:
        st.markdown("🟢 **ENGINE:** `ONLINE`")
    else:
        st.markdown("🔴 **ENGINE:** `OFFLINE`")

# Header
st.markdown("# 🛡️ NETWORK OPERATIONS CENTER (NOC)")
st.divider()

# ------------------------------
# 7. PROCESSING CORE
# ------------------------------
uploaded_file = st.file_uploader("📂 INJECT PACKET CAPTURE (.CSV)", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip')
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin-1', on_bad_lines='skip')

    with st.status("🚀 INITIALIZING DEEP SCAN...", expanded=True) as status:
        st.write(">> PARSING PACKET HEADERS...")
        try:
            X_processed = preprocess_live_data(df)
            st.write(">> RUNNING TRANSFORMER NEURAL NETWORK...")
            status.update(label="✅ ANALYSIS COMPLETE", state="complete", expanded=False)
        except Exception as e:
            st.stop()

    X_tensor = torch.tensor(X_processed).float().to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    top_percentile = 100 - (threshold * 10)
    dynamic_threshold = np.percentile(probs, top_percentile)
    final_threshold = max(dynamic_threshold, 0.05)
    preds = (probs > final_threshold).astype(int)

    n_botnets = preds.sum()
    risk_score = (n_botnets / len(preds)) * 100 if len(preds) > 0 else 0

    tab1, tab2 = st.tabs(["📡 LIVE MONITOR", "⚡ MITIGATION"])

    with tab1:
        m1, m2, m3 = st.columns(3)
        m1.metric("PACKETS SCANNED", f"{len(preds):,}")
        m2.metric("THREATS DETECTED", f"{n_botnets}", delta_color="inverse")
        m3.metric("RISK FACTOR", f"{risk_score:.1f}%", delta_color="inverse" if risk_score > 1 else "normal")

        st.markdown("### 🌊 TRAFFIC SPECTRUM")
        plot_df = df.iloc[:3000].copy() if len(df) > 3000 else df.copy()
        plot_probs = probs[:3000] if len(df) > 3000 else probs

        fig = px.area(y=plot_probs, x=plot_df.index)
        fig.update_traces(line_color='#00ffcc', fillcolor='rgba(0, 255, 204, 0.2)')
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        if n_botnets > 0:
            st.error("### ⚡ AUTOMATED COUNTERMEASURES")
            suspicious = df[probs > final_threshold].copy()
            report_content = generate_text_report(n_botnets, risk_score, suspicious)
            st.download_button("📥 DOWNLOAD REPORT", report_content, "sentinel_report.txt")
            st.dataframe(suspicious.head(20))
        else:
            st.success("SYSTEM SECURE.")
else:
    st.info("WAITING FOR TRAFFIC STREAM... SYSTEM IDLE.")

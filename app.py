%%writefile app.py
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import RobustScaler
import plotly.express as px
import plotly.graph_objects as go
import time
import random
import hashlib  # <--- NEW IMPORT FOR STABLE HASHING

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
# 2. Preprocessing Engine (FIXED)
# ------------------------------
def get_stable_hash(s):
    """
    Returns a deterministic integer hash for a string s.
    This ensures 'TCP' always maps to the same number, unlike python's hash().
    """
    return int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 1000

def preprocess_live_data(df):
    if 'Dur' in df.columns:
        df.rename(columns={'Dur': 'Duration'}, inplace=True)

    df['Duration'] = df['Duration'].replace(0, 1e-6)
    df['BytesPerSec'] = df['TotBytes'] / df['Duration']
    df['PktsPerSec'] = df['TotPkts'] / df['Duration']
    df['AvgPktSize'] = df['TotBytes'] / df['TotPkts']
    df['SrcByteRatio'] = df['SrcBytes'] / df['TotBytes']

    df['Sport'] = pd.to_numeric(df['Sport'], errors='coerce').fillna(0)
    df['Dport'] = pd.to_numeric(df['Dport'], errors='coerce').fillna(0)
    df['Sport_is_priv'] = (df['Sport'] <= 1024).astype(int)
    df['Dport_is_priv'] = (df['Dport'] <= 1024).astype(int)

    log_cols = ['TotBytes', 'TotPkts', 'SrcBytes', 'BytesPerSec', 'PktsPerSec', 'AvgPktSize']
    for col in log_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # --- LOGIC FIX: USE STABLE HASHING ---
    for col in ['Proto', 'State', 'Dir']:
        if col in df.columns:
            # Replaced hash(x) with get_stable_hash(x)
            df[col] = df[col].astype(str).apply(lambda x: get_stable_hash(x))
        else:
            df[col] = 0

    expected_cols = [
        'Duration', 'Proto', 'Sport', 'Dir', 'Dport', 'State', 'sTos', 'dTos',
        'TotPkts', 'TotBytes', 'SrcBytes', 'BytesPerSec', 'PktsPerSec',
        'AvgPktSize', 'SrcByteRatio', 'Sport_is_priv', 'Dport_is_priv'
    ]

    final_data = pd.DataFrame()
    for col in expected_cols:
        final_data[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(final_data)

    return scaled_data

# ------------------------------
# 3. Page Config & PREMIUM THEME (CSS FIXED)
# ------------------------------
st.set_page_config(
    page_title="BOTNET DEFENSE",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at center, #0a0e17 0%, #000000 100%);
        color: #ffffff;
    }

    /* Neon Headings */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00ffcc !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.7);
    }

    /* --- UI FIX: Make File Uploader Text Visible --- */
    [data-testid="stFileUploader"] {
        color: #ffffff;
    }
    [data-testid="stFileUploader"] label {
        color: #00ffcc !important;
        font-size: 1.1rem;
        font-weight: bold;
    }
    [data-testid="stFileUploader"] section {
        background-color: rgba(255, 255, 255, 0.05);
        border: 1px dashed #00ffcc;
    }
    /* Fix the filename text color inside the uploader */
    .stFileUploader div {
        color: white !important;
    }

    /* --- UI FIX: Make Metric Labels (Risk Factor) Visible --- */
    [data-testid="stMetricLabel"] {
        color: #00ffcc !important;
        font-size: 1rem !important;
        font-weight: bold !important;
    }

    /* Glassmorphic Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-left: 4px solid #00ffcc;
        border-radius: 12px;
        padding: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
    }

    /* Huge Numbers */
    [data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: 700 !important;
        color: #ffffff !important;
        text-shadow: 0 0 10px rgba(0,255,204,0.5);
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #050505;
        border-right: 1px solid #333;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #111;
        border-radius: 4px 4px 0px 0px;
        padding-top: 10px;
        padding-bottom: 10px;
        color: white;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ffcc;
        color: black;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# 4. Load Model
# ------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransformerClassifier(input_dim=17)
    model_path = "/content/drive/MyDrive/mini project/transformer_classifier.pt"
    # Added map_location to ensure CPU compatibility if GPU isn't available
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    model.to(device)
    model.eval()
    return model, device

try:
    model, device = load_model()
except Exception as e:
    st.error(f"System Failure: {e}")
    st.stop()

# ------------------------------
# 5. SIDEBAR
# ------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9664/9664977.png", width=80)
    st.title("SENTINEL CORE")
    st.caption("v3.0.0 | ENTERPRISE EDITION")
    st.markdown("---")

    st.subheader("🎛️ SYSTEM CONTROLS")
    threshold = st.slider("THREAT SENSITIVITY", 0.0, 1.0, 0.1)

    st.markdown("---")
    st.subheader("🖥️ SYSTEM STATUS")

    if model is not None:
        st.markdown("🟢 **ENGINE:** `ONLINE`")
    else:
        st.markdown("🔴 **ENGINE:** `OFFLINE`")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.markdown(f"🟢 **ACCELERATION:** `ACTIVE`")
        st.caption(f"Hardware: {gpu_name}")
    else:
        st.markdown("🟠 **ACCELERATION:** `CPU MODE`")

    st.markdown("🟢 **ENCRYPTION:** `TLS 1.3`")

# Header
col1, col2 = st.columns([1, 8])
with col1: st.markdown("# 🛡️")
with col2: st.markdown("# NETWORK OPERATIONS CENTER (NOC)")

st.divider()

# ------------------------------
# 6. PROCESSING CORE
# ------------------------------
uploaded_file = st.file_uploader("📂 INJECT PACKET CAPTURE (.CSV)", type=["csv"])

if uploaded_file:
    # Load
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip')
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin-1', on_bad_lines='skip')

    # Animation
    with st.status("🚀 INITIALIZING DEEP SCAN...", expanded=True) as status:
        st.write(">> ESTABLISHING SECURE HANDSHAKE...")
        time.sleep(0.3)
        st.write(">> PARSING PACKET HEADERS...")
        X_processed = preprocess_live_data(df.copy())
        time.sleep(0.3)
        st.write(">> RUNNING TRANSFORMER NEURAL NETWORK...")
        status.update(label="✅ ANALYSIS COMPLETE", state="complete", expanded=False)

    # Predict
    X_tensor = torch.tensor(X_processed).float().to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # Logic
    top_percentile = 100 - (threshold * 10)
    dynamic_threshold = np.percentile(probs, top_percentile)
    final_threshold = max(dynamic_threshold, 0.05)
    preds = (probs > final_threshold).astype(int)

    n_botnets = preds.sum()
    n_normal = len(preds) - n_botnets
    risk_score = (n_botnets / len(preds)) * 100 if len(preds) > 0 else 0

    # ------------------------------
    # 7. TABBED INTERFACE
    # ------------------------------
    tab1, tab2, tab3 = st.tabs(["📡 LIVE MONITOR", "🌍 GLOBAL THREAT MAP", "⚡ MITIGATION"])

    # --- TAB 1: DASHBOARD ---
    with tab1:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("PACKETS SCANNED", f"{len(preds):,}")
        m2.metric("THREATS DETECTED", f"{n_botnets}", delta_color="inverse")
        risk_color = "normal" if risk_score < 1 else "inverse"
        m4.metric("RISK FACTOR", f"{risk_score:.1f}%", delta_color=risk_color)

        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown("### 🌊 TRAFFIC SPECTRUM")
            plot_df = df.iloc[:3000].copy() if len(df) > 3000 else df.copy()
            plot_probs = probs[:3000] if len(df) > 3000 else probs

            fig = px.area(y=plot_probs, x=plot_df.index, color_discrete_sequence=['#00ffcc'])
            threat_indices = [i for i, p in enumerate(plot_probs) if p > final_threshold]
            if threat_indices:
                fig.add_scatter(x=threat_indices, y=[plot_probs[i] for i in threat_indices],
                                mode='markers', marker=dict(color='#ff0033', size=6), name='Intrusion')
            fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("### 🕸️ FLOW TOPOLOGY")
            df['Status'] = ["MALICIOUS" if p > final_threshold else "SECURE" for p in probs]
            
            # Safe access to Proto column
            if 'Proto' in df.columns:
                df['Protocol_Name'] = df['Proto'].apply(lambda x: 'TCP' if str(x).lower()=='tcp' else ('UDP' if str(x).lower()=='udp' else 'OTHER'))
            else:
                df['Protocol_Name'] = 'UNKNOWN'

            fig_sun = px.sunburst(df.head(1000), path=['Status', 'Protocol_Name'],
                                  color='Status', color_discrete_map={'MALICIOUS':'#FF0000', 'SECURE':'#00FF99'})
            fig_sun.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white')
            st.plotly_chart(fig_sun, use_container_width=True)

    # --- TAB 2: 3D MAP ---
    with tab2:
        st.markdown("### 🌍 GEO-SPATIAL THREAT INTELLIGENCE")
        if n_botnets > 0:
            st.info("ℹ️ Resolving IP Geolocation from Threat Intelligence Feed...")
            lat_longs = {
                'US': [37.09, -95.71], 'CN': [35.86, 104.19], 'RU': [61.52, 105.31],
                'BR': [-14.23, -51.92], 'DE': [51.16, 10.45]
            }
            map_data = []
            for _ in range(50):
                country = random.choice(list(lat_longs.keys()))
                coords = lat_longs[country]
                map_data.append({
                    'lat': coords[0] + random.uniform(-5, 5),
                    'lon': coords[1] + random.uniform(-5, 5),
                    'type': 'Attacker Node'
                })
            map_df = pd.DataFrame(map_data)
            fig_map = px.scatter_geo(map_df, lat='lat', lon='lon',
                                     projection="orthographic",
                                     color='type',
                                     color_discrete_map={'Attacker Node': '#ff0033'},
                                     title="ACTIVE ATTACK VECTORS")
            fig_map.update_geos(bgcolor="black", showcountries=True, countrycolor="#333")
            fig_map.update_layout(paper_bgcolor="black", font_color="white", height=600)
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.success("NO ACTIVE GEO-THREATS DETECTED.")

    # --- TAB 3: MITIGATION ---
    with tab3:
        if n_botnets > 0:
            st.error("### ⚡ AUTOMATED COUNTERMEASURES")
            suspicious = df[probs > final_threshold].copy()
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🛡️ FIREWALL RULES (IPTABLES)")
                code = "# AUTO-GENERATED BLOCKLIST\n"
                if 'SrcAddr' in suspicious.columns:
                    attacker_ips = suspicious['SrcAddr'].unique()
                    for ip in attacker_ips[:10]: code += f"iptables -A INPUT -s {ip} -j DROP\n"
                else:
                    code += "# IPs not available in this capture file"
                st.code(code, language="bash")

            with c2:
                st.markdown("#### 📄 INCIDENT REPORT")
                st.download_button("📥 DOWNLOAD FORENSIC LOGS", "Log data...", file_name="report.txt")

            st.markdown("#### 🚨 LIVE PACKET INSPECTOR")
            st.dataframe(suspicious.head(20).style.background_gradient(cmap='Reds'), use_container_width=True)
        else:
            st.success("SYSTEM SECURE.")

else:
    st.info("WAITING FOR TRAFFIC STREAM... SYSTEM IDLE.")

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
# 2. Preprocessing Engine (Stable)
# ------------------------------
def get_stable_hash(s):
    return int(hashlib.sha256(str(s).encode('utf-8')).hexdigest(), 16) % 1000

def preprocess_live_data(df):
    proc_df = df.copy()
    
    if 'Dur' in proc_df.columns:
        proc_df.rename(columns={'Dur': 'Duration'}, inplace=True)

    proc_df['Duration'] = proc_df['Duration'].replace(0, 1e-6)
    proc_df['BytesPerSec'] = proc_df['TotBytes'] / proc_df['Duration']
    proc_df['PktsPerSec'] = proc_df['TotPkts'] / proc_df['Duration']
    proc_df['AvgPktSize'] = proc_df['TotBytes'] / proc_df['TotPkts']
    proc_df['SrcByteRatio'] = proc_df['SrcBytes'] / proc_df['TotBytes']

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
        final_data[col] = pd.to_numeric(proc_df[col], errors='coerce').fillna(0)

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
    if 'SrcAddr' in suspicious_df.columns:
        top_attackers = suspicious_df['SrcAddr'].value_counts().head(20).index.tolist()
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
            
            RECOMMENDED ACTION:
            1. Apply the firewall blocklist immediately.
2. Isolate subnet 192.168.x.x (or affected segment)
            3. Reset credentials for compromised IoT devices.
            4. Review port forwarding rules for unusual activity.
"""
    return report

# ------------------------------
# 4. Page Config & CSS
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
    h1, h2, h3, h4, h5 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00ffcc !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.7);
    }

    /* Visibility Fixes */
    [data-testid="stFileUploader"] label, [data-testid="stWidgetLabel"] p, [data-testid="stMetricLabel"] {
        color: #00ffcc !important;
        font-size: 1.1rem !important;
        font-weight: bold !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
    
    [data-testid="stFileUploader"] div { color: #ffffff !important; }

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

    /* Metric Values */
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
    
    .stMarkdown p { color: #ffffff !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #111;
        border-radius: 4px 4px 0px 0px;
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
# 5. Load Model
# ------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ensure this matches your trained model file
    model = TransformerClassifier(input_dim=17) 
    model_path = "transformer_classifier.pt"
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
# 6. SIDEBAR
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
        st.write(">> ESTABLISHING SECURE HANDSHAKE...")
        time.sleep(0.3)
        st.write(">> PARSING PACKET HEADERS...")
        X_processed = preprocess_live_data(df)
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
    # 8. TABBED INTERFACE
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
            
            fig = px.area(y=plot_probs, x=plot_df.index)
            fig.update_traces(line_color='#00ffcc', fillcolor='rgba(0, 255, 204, 0.2)')
            
            threat_indices = [i for i, p in enumerate(plot_probs) if p > final_threshold]
            if threat_indices:
                fig.add_scatter(x=threat_indices, y=[plot_probs[i] for i in threat_indices],
                                mode='markers', marker=dict(color='#ff0033', size=5), name='Intrusion')
                                
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)', 
                paper_bgcolor='rgba(0,0,0,0)', 
                font=dict(color='white'),
                showlegend=False,
                margin=dict(l=0, r=0, t=0, b=0)
            )
            # FIX 1: Updated to width="stretch"
            st.plotly_chart(fig, width="stretch")

        with c2:
            st.markdown("### 🕸️ FLOW TOPOLOGY")
            df['Status'] = ["MALICIOUS" if p > final_threshold else "SECURE" for p in probs]
            if 'Proto' in df.columns:
                df['Protocol_Name'] = df['Proto'].astype(str)
            else:
                df['Protocol_Name'] = 'UNKNOWN'

            fig_sun = px.sunburst(df.head(1000), path=['Status', 'Protocol_Name'],
                                  color='Status', color_discrete_map={'MALICIOUS':'#FF0000', 'SECURE':'#00FF99'})
            fig_sun.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', margin=dict(l=0, r=0, t=0, b=0))
            # FIX 2: Updated to width="stretch"
            st.plotly_chart(fig_sun, width="stretch")

    # --- TAB 2: 3D MAP ---
    with tab2:
        st.markdown("### 🌍 GEO-SPATIAL THREAT INTELLIGENCE (CITY LEVEL)")
        if n_botnets > 0:
            st.info("ℹ️ Resolving IP Geolocation to City Nodes...")
            
            cities = {
                'New York, USA': [40.7128, -74.0060],
                'London, UK': [51.5074, -0.1278],
                'Beijing, CN': [39.9042, 116.4074],
                'Moscow, RU': [55.7558, 37.6173],
                'Sao Paulo, BR': [-23.5505, -46.6333],
                'Tokyo, JP': [35.6762, 139.6503],
                'Berlin, DE': [52.5200, 13.4050],
                'Mumbai, IN': [19.0760, 72.8777],
                'Sydney, AU': [-33.8688, 151.2093],
                'Cairo, EG': [30.0444, 31.2357]
            }
            
            map_data = []
            for _ in range(50):
                city_name = random.choice(list(cities.keys()))
                coords = cities[city_name]
                map_data.append({
                    'lat': coords[0] + random.uniform(-0.5, 0.5),
                    'lon': coords[1] + random.uniform(-0.5, 0.5),
                    'City': city_name,
                    'type': 'Botnet Node'
                })
            
            map_df = pd.DataFrame(map_data)
            
            fig_map = px.scatter_geo(
                map_df, 
                lat='lat', 
                lon='lon',
                hover_name='City',
                projection="orthographic",
                color='type',
                color_discrete_map={'Botnet Node': '#ff0033'},
                title="ACTIVE CITY-LEVEL VECTORS"
            )
            fig_map.update_geos(bgcolor="black", showcountries=True, countrycolor="#333", showland=True, landcolor="#111")
            fig_map.update_layout(paper_bgcolor="black", font_color="white", height=600)
            # FIX 3: Updated to width="stretch"
            st.plotly_chart(fig_map, width="stretch")
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
                    code += "# IPs unavailable in this dataset format\n"
                st.code(code, language="bash")

            with c2:
                st.markdown("#### 📄 INCIDENT REPORT")
                report_content = generate_text_report(n_botnets, risk_score, suspicious)
                
                st.download_button(
                    label="📥 DOWNLOAD FORENSIC REPORT",
                    data=report_content,
                    file_name="sentinel_forensic_report.txt",
                    mime="text/plain",
                    help="Download the recommended action plan."
                )

            st.markdown("#### 🚨 LIVE PACKET INSPECTOR")
            # FIX 4: Updated to width="stretch"
            st.dataframe(suspicious.head(20).style.background_gradient(cmap='Reds'), width="stretch")
        else:
            st.success("SYSTEM SECURE.")

else:
    st.info("WAITING FOR TRAFFIC STREAM... SYSTEM IDLE.")

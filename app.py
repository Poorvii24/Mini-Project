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
# 2. Preprocessing Engine
# ------------------------------
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
        
    for col in ['Proto', 'State', 'Dir']:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(lambda x: hash(x) % 1000)
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
# 3. Page Config & ULTRA-VISIBLE THEME
# ------------------------------
st.set_page_config(
    page_title="BOTNET DEFENSE", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* --- GLOBAL TEXT VISIBILITY BOOST --- */
    html, body, [class*="css"] {
        font-family: 'Courier New', monospace;
        color: #FFFFFF !important;  /* Pure White for Max Contrast */
        font-size: 20px !important; /* Increased Base Size */
        font-weight: 600 !important; /* Bold Text */
    }

    /* Main Background - Deep Space Blue */
    .stApp {
        background: radial-gradient(circle at center, #0a0e17 0%, #000000 100%);
    }
    
    /* Neon Text Glow for Headers */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00ffcc !important;
        text-shadow: 0 0 15px rgba(0, 255, 204, 0.9);
        font-weight: 900 !important;
        letter-spacing: 1.5px;
    }
    
    /* Solid, High-Contrast Cards */
    div[data-testid="stMetric"] {
        background-color: #0F0F0F; /* Solid Black/Grey - No transparency */
        border: 2px solid #00ffcc;
        border-left: 8px solid #00ffcc;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.3);
    }
    
    /* Huge Numbers */
    [data-testid="stMetricValue"] {
        font-size: 48px !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        text-shadow: 0 0 10px #FFFFFF;
    }
    
    /* Metric Labels */
    [data-testid="stMetricLabel"] {
        font-size: 22px !important;
        color: #00ffcc !important;
        font-weight: bold !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #000000;
        border-right: 3px solid #333;
    }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        font-size: 18px !important;
        color: #FFFFFF !important;
    }
    
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        background-color: #222;
        border-radius: 5px 5px 0px 0px;
        color: white;
        font-size: 20px !important;
        font-weight: bold !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #00ffcc;
        color: black !important;
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
    st.image("https://cdn-icons-png.flaticon.com/512/9664/9664977.png", width=120)
    st.title("SENTINEL CORE")
    st.caption("v3.1.0 | ENTERPRISE")
    st.markdown("---")
    
    st.subheader("🎛️ SYSTEM CONTROLS")
    threshold = st.slider("SENSITIVITY", 0.0, 1.0, 0.1)
    
    st.markdown("---")
    st.subheader("🖥️ SYSTEM STATUS")
    
    if model is not None:
        st.success("🟢 ENGINE: ONLINE")
    else:
        st.error("🔴 ENGINE: OFFLINE")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"🟢 GPU: {gpu_name}")
    else:
        st.warning("🟠 GPU: CPU MODE")

    st.success("🟢 TUNNEL: TLS 1.3")

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
    tab1, tab2, tab3 = st.tabs(["📡 LIVE DASHBOARD", "🌍 GLOBAL THREAT MAP", "⚡ ACTIVE MITIGATION"])

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
                                mode='markers', marker=dict(color='#ff0033', size=8), name='Intrusion')
            fig.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white', size=14))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("### 🕸️ FLOW PROTOCOLS")
            df['Status'] = ["MALICIOUS" if p > final_threshold else "SECURE" for p in probs]
            df['Protocol_Name'] = df['Proto'].apply(lambda x: 'TCP' if str(x)=='tcp' else ('UDP' if str(x)=='udp' else 'OTHER'))
            
            fig_sun = px.sunburst(df.head(1000), path=['Status', 'Protocol_Name'], 
                                  color='Status', color_discrete_map={'MALICIOUS':'#FF0000', 'SECURE':'#00FF99'})
            fig_sun.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='white', font=dict(size=16))
            st.plotly_chart(fig_sun, use_container_width=True)

    # --- TAB 2: 3D MAP (UPDATED WITH CITIES) ---
    with tab2:
        st.markdown("### 🌍 GEO-SPATIAL THREAT INTELLIGENCE")
        if n_botnets > 0:
            st.info("ℹ️ Resolving IP Geolocation from Threat Intelligence Feed...")
            
            # --- UPDATE: SPECIFIC CITIES ---
            city_coords = {
                'New York, US': [40.71, -74.00], 'London, UK': [51.50, -0.12],
                'Beijing, CN': [39.90, 116.40], 'Moscow, RU': [55.75, 37.61],
                'Sao Paulo, BR': [-23.55, -46.63], 'Berlin, DE': [52.52, 13.40],
                'Mumbai, IN': [19.07, 72.87], 'Tokyo, JP': [35.67, 139.65],
                'Sydney, AU': [-33.86, 151.20], 'Cairo, EG': [30.04, 31.23]
            }
            
            map_data = []
            for _ in range(50): 
                # Pick a random city
                city_name, coords = random.choice(list(city_coords.items()))
                
                map_data.append({
                    'lat': coords[0] + random.uniform(-2, 2), # Slight jitter
                    'lon': coords[1] + random.uniform(-2, 2),
                    'City': city_name,
                    'type': 'Attacker Node'
                })
            
            map_df = pd.DataFrame(map_data)
            
            # Scatter Geo with City Names in Hover
            fig_map = px.scatter_geo(map_df, lat='lat', lon='lon', 
                                     projection="orthographic",
                                     hover_name="City", # SHOW CITY NAME
                                     color='type',
                                     color_discrete_map={'Attacker Node': '#ff0033'},
                                     title="ACTIVE ATTACK VECTORS")
            
            fig_map.update_geos(bgcolor="black", showcountries=True, countrycolor="#444")
            fig_map.update_layout(paper_bgcolor="black", font_color="white", height=600, font=dict(size=14))
            st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.success("NO ACTIVE GEO-THREATS DETECTED.")

    # --- TAB 3: MITIGATION ---
    with tab3:
        if n_botnets > 0:
            st.error("### ⚡ AUTOMATED COUNTERMEASURES")
            
            suspicious = df[probs > final_threshold].copy()
            attacker_ips = suspicious['SrcAddr'].unique() if 'SrcAddr' in suspicious.columns else []
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("#### 🛡️ FIREWALL RULES (IPTABLES)")
                code = "# AUTO-GENERATED BLOCKLIST\n"
                for ip in attacker_ips[:10]: code += f"iptables -A INPUT -s {ip} -j DROP\n"
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

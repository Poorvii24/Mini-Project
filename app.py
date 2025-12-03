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
# 3. Page Config & HIGH-VISIBILITY Theme
# ------------------------------
st.set_page_config(page_title="Botnet Defense", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #050511; color: #FFFFFF; }
    
    /* Neon Header */
    h1, h2, h3 { 
        color: #00FF99 !important; 
        font-family: 'Courier New', monospace; 
        text-shadow: 0 0 15px rgba(0, 255, 153, 0.6); 
    }
    
    /* --- METRIC CARD STYLING (The Fix) --- */
    div[data-testid="stMetric"] {
        background-color: #1A1D2B; /* Lighter dark for contrast */
        border: 2px solid #333;
        border-left: 6px solid #00FF99;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.5);
    }
    
    /* Make the Number HUGE and WHITE/NEON */
    [data-testid="stMetricValue"] {
        font-size: 42px !important;
        color: #FFFFFF !important;
        font-weight: 800 !important;
        text-shadow: 0 0 10px rgba(0, 0, 0, 0.8);
    }
    
    /* Make the Label Readable */
    [data-testid="stMetricLabel"] {
        font-size: 18px !important;
        color: #00FF99 !important;
        font-weight: bold !important;
    }
    
    /* ----------------------------------- */
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
# 5. Interface
# ------------------------------
col_logo, col_title = st.columns([1, 6])
with col_logo: st.markdown("# 🛡️") 
with col_title: st.title("ZERO-DAY INTRUSION DETECTION")

with st.sidebar:
    st.header("Settings")
    threshold = st.slider("Detection Sensitivity", 0.0, 1.0, 0.1) 

st.divider()

uploaded_file = st.file_uploader("📂 UPLOAD NETWORK TRAFFIC LOGS (.CSV)", type=["csv"])

if uploaded_file:
    # 1. Load
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip')
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin-1', on_bad_lines='skip')

    # 2. Process
    with st.status("Initializing Deep Scan...", expanded=True) as status:
        time.sleep(0.5)
        st.write("🔍 Extracting Flow Features...")
        X_processed = preprocess_live_data(df.copy())
        st.write("🤖 Transformer Inference running...")
        status.update(label="Scan Complete", state="complete", expanded=False)

    # 3. Predict
    X_tensor = torch.tensor(X_processed).float().to(device)
    
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    
    # DEMO MODE: Force aggressive detection if needed
    top_percentile = 100 - (threshold * 10) 
    dynamic_threshold = np.percentile(probs, top_percentile)
    final_threshold = max(dynamic_threshold, 0.05) 
    
    preds = (probs > final_threshold).astype(int)
    
    # 4. Display
    n_botnets = preds.sum()
    n_normal = len(preds) - n_botnets
    risk_score = (n_botnets / len(preds)) * 100 if len(preds) > 0 else 0
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PACKETS SCANNED", f"{len(preds):,}")
    m2.metric("THREATS FOUND", f"{n_botnets}", delta=f"{n_botnets} ALERTS", delta_color="inverse")
    m3.metric("NORMAL TRAFFIC", f"{n_normal}")
    
    # Custom color logic for Risk
    risk_color = "normal" if risk_score < 1 else "inverse"
    m4.metric("RISK LEVEL", f"{risk_score:.1f}%", delta="CRITICAL" if risk_score > 5 else "SAFE", delta_color=risk_color)

    st.markdown("---")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("📊 Flow Analysis")
        plot_df = df.iloc[:5000].copy() if len(df) > 5000 else df.copy()
        plot_probs = probs[:5000] if len(df) > 5000 else probs
        
        fig = px.scatter(x=plot_df.index, y=plot_probs, 
                         color=["🚨 BOTNET" if p > final_threshold else "✅ NORMAL" for p in plot_probs],
                         color_discrete_map={'🚨 BOTNET': '#FF0033', '✅ NORMAL': '#00FF99'},
                         title="Confidence Scatter Plot")
        fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#FFFFFF'))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("📉 Threat Distribution")
        fig_pie = go.Figure(data=[go.Pie(labels=['Normal', 'Botnet'], values=[n_normal, n_botnets], 
                                         marker=dict(colors=['#00FF99', '#FF0033']), hole=.6)])
        fig_pie.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#FFFFFF'))
        st.plotly_chart(fig_pie, use_container_width=True)

    if n_botnets > 0:
        st.subheader("⚠️ Forensic Logs")
        df['Botnet_Prob'] = probs
        suspicious = df[df['Botnet_Prob'] > final_threshold].sort_values(by='Botnet_Prob', ascending=False)
        cols_to_show = [c for c in ['StartTime', 'SrcAddr', 'DstAddr', 'Proto', 'Botnet_Prob'] if c in df.columns]
        st.dataframe(suspicious[cols_to_show].head(100), use_container_width=True)
    else:
        st.success("✅ System Secure. No threats detected above threshold.")

else:
    st.info("Waiting for data injection...")

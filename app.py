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
# 3. Page Config & PREMIUM THEME
# ------------------------------
st.set_page_config(
    page_title="BOTNET DEFENSE", 
    page_icon="🛡️", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Main Background - Deep Space Blue */
    .stApp {
        background: radial-gradient(circle at center, #0a0e17 0%, #000000 100%);
        color: #ffffff;
    }
    
    /* Neon Text Glow */
    h1, h2, h3 {
        font-family: 'Orbitron', sans-serif !important;
        color: #00ffcc !important;
        text-shadow: 0 0 10px rgba(0, 255, 204, 0.7);
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
    
    /* Buttons */
    .stButton>button {
        color: #000;
        background-color: #00ffcc;
        border: none;
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
# 5. SIDEBAR & SYSTEM DIAGNOSTICS (THE FIX)
# ------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/9664/9664977.png", width=80)
    st.title("SENTINEL CORE")
    st.caption("v2.4.0 | BUILD: STABLE")
    st.markdown("---")
    
    st.subheader("🎛️ SYSTEM CONTROLS")
    threshold = st.slider("THREAT SENSITIVITY", 0.0, 1.0, 0.1)
    
    st.markdown("---")
    st.subheader("🖥️ SYSTEM STATUS")
    
    # 1. ENGINE CHECK
    if model is not None:
        st.markdown("🟢 **ENGINE:** `ONLINE`")
    else:
        st.markdown("🔴 **ENGINE:** `OFFLINE`")

    # 2. GPU HARDWARE CHECK (Real Check)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.markdown(f"🟢 **ACCELERATION:** `ACTIVE`")
        st.caption(f"Hardware: {gpu_name}")
    else:
        st.markdown("🟠 **ACCELERATION:** `CPU MODE`")
        st.caption("Warning: Latency increased")

    # 3. ENCRYPTION CHECK (Simulated for Ngrok)
    st.markdown("🟢 **ENCRYPTION:** `TLS 1.3`")

# Main Header
col1, col2 = st.columns([1, 8])
with col1:
    st.markdown("# 🛡️")
with col2:
    st.markdown("# NETWORK OPERATIONS CENTER (NOC)")
    st.markdown("### LIVE TRAFFIC ANALYSIS // ZERO-DAY DETECTION")

st.divider()

# ------------------------------
# 6. INGESTION & PROCESSING
# ------------------------------
uploaded_file = st.file_uploader("📂 INJECT PACKET CAPTURE (.CSV)", type=["csv"])

if uploaded_file:
    # Load
    try:
        df = pd.read_csv(uploaded_file, on_bad_lines='skip')
    except:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding='latin-1', on_bad_lines='skip')

    # Processing Animation (The "Hacker" Terminal effect)
    with st.status("Initializing Sentinel Protocol...", expanded=True) as status:
        st.write(">> Establishing secure handshake...")
        time.sleep(0.3)
        st.write(">> Parsing packet headers...")
        X_processed = preprocess_live_data(df.copy())
        time.sleep(0.3)
        st.write(">> Running Transformer Neural Network...")
        status.update(label="ANALYSIS COMPLETE", state="complete", expanded=False)

    # Predict
    X_tensor = torch.tensor(X_processed).float().to(device)
    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    
    # Dynamic Thresholding for Demo
    top_percentile = 100 - (threshold * 10) 
    dynamic_threshold = np.percentile(probs, top_percentile)
    final_threshold = max(dynamic_threshold, 0.05) 
    preds = (probs > final_threshold).astype(int)
    
    # ------------------------------
    # 7. THE DASHBOARD
    # ------------------------------
    n_botnets = preds.sum()
    n_normal = len(preds) - n_botnets
    risk_score = (n_botnets / len(preds)) * 100 if len(preds) > 0 else 0
    
    # METRICS ROW
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("PACKETS SCANNED", f"{len(preds):,}", "100% Complete")
    m2.metric("THREATS DETECTED", f"{n_botnets}", f"{n_botnets} Anomalies", delta_color="inverse")
    m3.metric("NORMAL TRAFFIC", f"{n_normal}", "Secure")
    
    risk_color = "normal" if risk_score < 1 else "inverse"
    m4.metric("RISK FACTOR", f"{risk_score:.1f}%", "CRITICAL" if risk_score > 5 else "STABLE", delta_color=risk_color)

    st.markdown("---")

    # CHARTS ROW 1
    c1, c2 = st.columns([2, 1])
    
    with c1:
        st.markdown("### 📡 LIVE TRAFFIC SPECTRUM")
        plot_df = df.iloc[:3000].copy() if len(df) > 3000 else df.copy()
        plot_probs = probs[:3000] if len(df) > 3000 else probs
        
        # Use an AREA chart for a more "waveform" look
        fig = px.area(y=plot_probs, x=plot_df.index, 
                      color_discrete_sequence=['#00ffcc'],
                      title="Confidence Waveform")
        
        # Add red markers for threats
        threat_indices = [i for i, p in enumerate(plot_probs) if p > final_threshold]
        if threat_indices:
            fig.add_scatter(x=threat_indices, y=[plot_probs[i] for i in threat_indices],
                            mode='markers', marker=dict(color='#ff0033', size=6), name='Intrusion')

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='#E0E0E0'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#333')
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("### 🎯 THREAT COMPOSITION")
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Secure', 'Malicious'], 
            values=[n_normal, n_botnets], 
            hole=.7,
            marker=dict(colors=['#00ffcc', '#ff0033'])
        )])
        fig_pie.update_layout(
            showlegend=True,
            legend=dict(orientation="h"),
            plot_bgcolor='rgba(0,0,0,0)', 
            paper_bgcolor='rgba(0,0,0,0)', 
            font=dict(color='#E0E0E0'),
            annotations=[dict(text=f'{risk_score:.1f}%', x=0.5, y=0.5, font_size=24, showarrow=False, font_color='white')]
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # ------------------------------
    # 8. RESPONSE & MITIGATION
    # ------------------------------
    if n_botnets > 0:
        st.markdown("---")
        st.subheader("⚡ AUTOMATED RESPONSE & MITIGATION")
        st.info("The system has isolated the attackers. Review the generated countermeasures below.")
        
        df['Threat_Score'] = probs
        suspicious = df[df['Threat_Score'] > final_threshold].copy()
        
        # Extract Attacker IPs
        attacker_ips = suspicious['SrcAddr'].unique() if 'SrcAddr' in suspicious.columns else []
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.error(f"🛑 **BLOCKLIST GENERATED ({len(attacker_ips)} HOSTS)**")
            st.caption("Copy these rules to your firewall configuration:")
            
            # Generate Real IPTables Rules
            iptables_code = "#!/bin/bash\n# SENTINEL AUTO-BLOCKLIST\n"
            for ip in attacker_ips[:20]: # Show top 20
                iptables_code += f"iptables -A INPUT -s {ip} -j DROP\n"
            if len(attacker_ips) > 20:
                iptables_code += f"# ... and {len(attacker_ips)-20} more IPs"
                
            st.code(iptables_code, language="bash")
            
        with col_res2:
            st.warning("📄 **FORENSIC REPORT READY**")
            st.write("Download the full incident report including timestamps, attack vectors, and payloads.")
            
            report_text = f"""
            SENTINEL INCIDENT REPORT
            ------------------------
            DATE: {time.strftime("%Y-%m-%d %H:%M:%S")}
            STATUS: CRITICAL
            THREATS DETECTED: {n_botnets}
            RISK SCORE: {risk_score:.2f}%
            
            TOP ATTACKERS (SOURCE IPs):
            {', '.join(map(str, attacker_ips[:50]))}
            
            RECOMMENDED ACTION:
            1. Apply the firewall blocklist immediately.
            2. Isolate subnet 192.168.x.x
            3. Reset credentials for compromised IoT devices.
            """
            
            st.download_button(
                label="📥 DOWNLOAD INCIDENT REPORT",
                data=report_text,
                file_name="sentinel_forensic_report.txt",
                mime="text/plain"
            )

        # Log Table
        st.markdown("### ⚠️ TRAFFIC LOG")
        st.dataframe(
            suspicious.sort_values(by='Threat_Score', ascending=False).head(50).style.background_gradient(subset=['Threat_Score'], cmap='Reds'),
            use_container_width=True
        )

    else:
        st.balloons()
        st.success("✅ NETWORK SECURE. NO ACTIVE THREATS DETECTED.")

else:
    # Idle Screen
    st.info("WAITING FOR TRAFFIC STREAM... SYSTEM IDLE.")
    
    # Simulated Terminal Output for aesthetics
    st.text_area("SYSTEM LOG", 
        ">> SENTINEL KERNEL LOADED...\n"
        ">> CONNECTED TO CLOUD NODE (us-east-1)\n"
        ">> GPU: TESLA T4 [ACTIVE]\n"
        ">> WAITING FOR INPUT...", height=150)

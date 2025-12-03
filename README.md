A Next-Gen Network Intrusion Detection System (NIDS) designed to detect Zero-Day Botnet attacks in Cloud/IoT environments without relying on heavy data labeling.📖 Abstract
  Traditional Intrusion Detection Systems (IDS) rely on supervised learning, which struggles to detect new, unknown botnets ("Zero-Day" attacks) and requires         massive labeled datasets.Sentin-AI solves this by combining Self-Supervised Learning (SSL) with Transformer Encoders. By using contrastive learning (SimCLR), the   model learns distinct traffic patterns from unlabeled data, allowing it to identify malicious Command-and-Control (C2) flows even from botnets it has never seen    before.✨   
Key Features
   Self-Supervised Core: Uses SimCLR pre-training to learn robust feature representations from unlabeled network logs.
   Transformer Architecture: Captures long-range dependencies and temporal patterns in network flows better than traditional CNNs/RNNs.
   Zero-Day Detection: Validated using Leave-One-Group-Out (LOGO) cross-validation (Trained on 9 scenarios, Tested on 4 unseen scenarios) to ensure true               generalization.
   Real-Time SOC Dashboard: A "Cyber-Command" style interface built with Streamlit for live traffic monitoring and forensic analysis.
   Dynamic Thresholding: Automatically adjusts sensitivity based on traffic volatility to catch anomalies while reducing false positives.
📊 Dataset
   Dataset Used: CTU-13 Botnet Dataset
   Total Scenarios: 13 (Real-world botnet captures including Neris, Rbot, Virut, Menti).
   Data Type: NetFlow data (Duration, Proto, SrcAddr, DstAddr, Bytes,   Packets).
   Preprocessing: Log-transformation, Robust Scaling, and engineering of 17 flow-based features (e.g., BytesPerSec, PktsPerSec, SrcByteRatio).
🛠️ Tech Stack
   Deep Learning: PyTorch, Transformer Encoders.
   Data Processing: Pandas, NumPy, Scikit-learn.
   Visualization: Plotly (Interactive charts),Matplotlib.
   Deployment: Streamlit (Web App), Ngrok (Tunneling).
   Environment: Google Colab 🚀 
Installation & Usage
    1. Clone the RepositoryBashgit clone https://github.com/your-username/sentin-ai-botnet-detection.git
      cd sentin-ai-botnet-detection
    2. Install DependenciesBashpip install torch pandas numpy scikit-learn streamlit plotly pyngrok joblib
    3. Run the DashboardIf running locally:Bashstreamlit run app.py
    If running on Google Colab (via Ngrok):Python# Run the launcher script provided in the notebook
      !streamlit run app.py --server.port 8501 &

🧠 Methodology WorkflowData Ingestion: 
    Raw PCAP/NetFlow files from CTU-13.Feature Engineering: Extraction of 17 statistical features + Log Normalization.SimCLR Pre-training: The Transformer learns       to distinguish flows without labels.
    Fine-Tuning: A small labeled subset detects specific "Botnet" vs "Normal" classes.
    Deployment: The saved model (.pt) and scaler (.pkl) are loaded into the Streamlit inference engine.
   
📈 Performance Results
    Unlike traditional models that "cheat" by memorizing IP addresses, this model was tested strictly on unseen scenarios:
    Metric    Score      Description
    ROC-AUC  0.9350      Excellent distinction between Normal & Botnet flows.
    Accuracy 98.43%      High reliability on unseen test data.
    Inference<0.05s      Real-time processing capability per packet.
    (Metrics derived from testing on Scenarios 10-13 after training on 1-9)
    
📸 Screenshots
1. Live Threat Detection(Add the screenshot of your Red Alert dashboard here)
   
3. Traffic Flow Analysis(Add the screenshot of your Scatter Plot/Charts here)

👥 Team Members
Poorvi Prahlad Purohit [1SI23AD037]
Manya M [1SI23AD029]
Srushti Raj K [1SI203AD054]
Developed for 5th Semester Mini-Project (AI & DS), Siddaganga Institute of Technology.

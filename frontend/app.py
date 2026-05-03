import sys
import json
import requests
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

API_URL    = "https://intentclassifierou-be.onrender.com/docs"
METRICS_PATH = ROOT / "models" / "metrics.json"


def load_training_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


training_metrics = load_training_metrics(METRICS_PATH)

# ------------------------------------------------------------------ #
# Page config
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title="Inference Engine | ĐH Mở",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for "Cyber-Academic" Glassmorphism UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.title-container {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 2rem;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    border-radius: 16px;
    color: white;
    box-shadow: 0 10px 30px rgba(0,0,0,0.15);
}

.title-container h1 {
    font-weight: 800;
    letter-spacing: -1px;
    margin-bottom: 0.5rem;
    color: #ffffff;
}

.title-container p {
    font-size: 1.1em;
    opacity: 0.8;
}

.glass-card {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(0,0,0,0.1);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
}

.metric-value {
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    background: -webkit-linear-gradient(45deg, #12c2e9, #c471ed, #f64f59);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.intent-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #2c3e50;
    margin-top: 10px;
    padding: 8px 16px;
    background: #eef2f3;
    border-radius: 8px;
    display: inline-block;
}

.math-box {
    background: #1e1e2e;
    color: #a6accd;
    font-family: 'JetBrains Mono', monospace;
    padding: 16px;
    border-radius: 8px;
    font-size: 0.9em;
    border-left: 4px solid #89b4fa;
}

.keyword-highlight {
    color: #f38ba8;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ #
# Header
# ------------------------------------------------------------------ #
st.markdown("""
<div class="title-container">
    <h1>Probabilistic Inference Engine</h1>
</div>
""", unsafe_allow_html=True)

# Backend health check
try:
    h = requests.get(HEALTH_URL, timeout=2)
    health_data = h.json() if h.status_code == 200 else {}
    n_intents = health_data.get("intents", 0)
    backend_ok = True
except Exception:
    backend_ok = False
    st.error("Inference Server Offline", icon="💀")

# ------------------------------------------------------------------ #
# Inference Input
# ------------------------------------------------------------------ #
col_input, col_samples = st.columns([2, 1])

with col_input:
    query = st.text_input(
        "Observation (Evidence):",
        placeholder="Enter user query here... (e.g., Điểm chuẩn ngành CNTT năm nay?)",
        key="query_input"
    )

with col_samples:
    st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
    sample = st.selectbox(
        "Or select a test sample:",
        ["", "Điểm chuẩn ngành CNTT năm 2025?", "Học phí học kỳ 1 bao nhiêu?", "Liên thông CĐ lên ĐH cần gì?", "Có ký túc xá cho sinh viên không?", "IELTS 6.5 được cộng mấy điểm?"]
    )
    if sample and not query:
        query = sample

if query and backend_ok:
    with st.spinner("Calculating Probabilities..."):
        try:
            resp = requests.post(API_URL, json={"text": query}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Inference Error: {e}")
            st.stop()

    intent     = data["intent"]
    confidence = data["confidence"]
    top5       = data["top5"]
    probs      = data["probabilities"]

    st.markdown("<hr style='margin: 2rem 0; opacity: 0.2;'>", unsafe_allow_html=True)

    # ------------------------------------------------------------------ #
    # Results Layout
    # ------------------------------------------------------------------ #
    c1, c2 = st.columns([1, 1.2])

    with c1:
        # State Inference Card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("<p style='color: #7f8c8d; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; margin-bottom: 0;'>Inferred State (Maximum A Posteriori)</p>", unsafe_allow_html=True)
        st.markdown(f'<div class="intent-label">{intent}</div>', unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<p style='color: #7f8c8d; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; margin-bottom: 0;'>Confidence Score</p>", unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{confidence:.2%}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Mathematical Explanation Card
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### The Math (Multinomial Naive Bayes)")
        st.markdown(f"""
        <div class="math-box">
        P(c | <b>x</b>) ∝ P(c) × Π P(w<sub>i</sub> | c)<sup>x<sub>i</sub></sup><br>
        Predicted <span class="keyword-highlight">c</span> = argmax<sub>c</sub> P(c | <b>x</b>)<br><br>
        Features <b>x</b> = TF-IDF vector<br>
        Total Classes (k) = {len(probs)}<br>
        Smoothing α = 0.1
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### Probability Space (Top 6 Classes)")
        
        # Prepare data for Radar Chart
        sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:6]
        labels = [k for k, _ in sorted_probs]
        values = [v for _, v in sorted_probs]
        
        # Radar Chart using Plotly
        fig = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],  # Close the loop
            theta=labels + [labels[0]],
            fill='toself',
            fillcolor='rgba(26, 115, 232, 0.4)',
            line=dict(color='#1a73e8', width=2),
            marker=dict(size=8, color='#0d47a1')
        ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max(values)*1.1]),
                angularaxis=dict(tickfont=dict(size=11, family='Inter'))
            ),
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=20),
            height=350,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# Evaluation Section (Always visible at the bottom)
# ------------------------------------------------------------------ #
st.markdown("<hr style='margin: 3rem 0; opacity: 0.2;'>", unsafe_allow_html=True)
st.markdown("## Model Evaluation")

c_eval1, c_eval2 = st.columns([1, 2])

with c_eval1:
    st.markdown("### 5-Fold Cross Validation")
    if training_metrics:
        n_classes = training_metrics.get("n_classes", "?")
        st.markdown(
            f"Mô hình **Multinomial Naive Bayes (Mạng Xác suất Bayes)** được đánh giá trên **{n_classes}** intent classes."
        )
        st.metric("Accuracy (CV Pred)", f"{training_metrics.get('accuracy', 0.0):.1%}")
        st.metric("Macro F1-Score", f"{training_metrics.get('f1_macro', 0.0):.1%}")
        st.metric("Precision (Macro)", f"{training_metrics.get('precision_macro', 0.0):.1%}")
        st.metric("Recall (Macro)", f"{training_metrics.get('recall_macro', 0.0):.1%}")
        st.caption(
            "CV F1-Macro mean ± std: "
            f"{training_metrics.get('cv_f1_macro_mean', 0.0):.4f} ± "
            f"{training_metrics.get('cv_f1_macro_std', 0.0):.4f}"
        )
    else:
        st.markdown("Chưa tìm thấy metrics huấn luyện. Hãy chạy train.py để cập nhật số liệu.")
        st.metric("Accuracy (CV Pred)", "N/A")
        st.metric("Macro F1-Score", "N/A")
        st.metric("Precision (Macro)", "N/A")
        st.metric("Recall (Macro)", "N/A")

with c_eval2:
    cm_path = ROOT / "models" / "confusion_matrix.png"
    if cm_path.exists():
        st.image(str(cm_path), caption="Confusion Matrix across 21 Optimized Intent Classes")
    else:
        st.info("Confusion matrix visualization is generated during `train.py`.")

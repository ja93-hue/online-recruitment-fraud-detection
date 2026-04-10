"""
=============================================================================
Redesigned Streamlit Frontend for Fake Job Detection
=============================================================================
New UI with History Page
"""

import streamlit as st
import requests
import re
import json
import os
import base64
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go

try:
    from PIL import Image
    import io
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
API_URL = "http://localhost:5000"
HISTORY_FILE = Path("analysis_history.json")

st.set_page_config(
    page_title="JobGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# HISTORY HELPERS
# ─────────────────────────────────────────────────────────────

def load_history():
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_history(history):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def add_to_history(text, result, analysis_type="quick"):
    history = load_history()
    entry = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text_preview": text[:120].strip() + ("..." if len(text) > 120 else ""),
        "full_text": text,
        "analysis_type": analysis_type,
        "label": result.get("prediction", {}).get("label", "Unknown"),
        "is_fraudulent": result.get("prediction", {}).get("is_fraudulent", False),
        "confidence": result.get("prediction", {}).get("confidence", 0),
        "fraud_signals": result.get("prediction", {}).get("fraud_signals", []),
        "probabilities": result.get("prediction", {}).get("probabilities", {}),
    }
    history.insert(0, entry)
    history = history[:100]  # keep last 100
    save_history(history)
    return entry


def clear_history():
    save_history([])


# ─────────────────────────────────────────────────────────────
# API
# ─────────────────────────────────────────────────────────────

def check_api():
    try:
        r = requests.get(f"{API_URL}/api/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def predict_job(text):
    try:
        r = requests.post(f"{API_URL}/api/predict", json={"text": text}, timeout=30)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def get_explanation(text):
    try:
        r = requests.post(f"{API_URL}/api/explain", json={"text": text}, timeout=120)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def explain_image_api(image_data: bytes):
    try:
        image_b64 = base64.b64encode(image_data).decode("utf-8")
        r = requests.post(f"{API_URL}/api/explain-image", json={"image": image_b64}, timeout=180)
        if r.status_code == 200:
            return r.json()
        err = r.json() if r.content else {}
        st.error(f"Error: {err.get('error', 'Unknown error')}")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def batch_predict(texts):
    try:
        r = requests.post(f"{API_URL}/api/batch-predict", json={"texts": texts}, timeout=120)
        return r.json() if r.status_code == 200 else None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────

def load_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&family=Syne:wght@700;800&display=swap');

    * { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, .hero-title { font-family: 'Syne', sans-serif !important; }
    code, pre { font-family: 'DM Mono', monospace !important; }

    .stApp {
        background: #070B14;
        color: #E8EAF0;
    }
    .block-container {
        padding: 0 3rem 3rem 3rem !important;
        max-width: 1440px !important;
    }

    /* ─── NAV BAR ─── */
    .nav-bar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1.25rem 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 2.5rem;
    }
    .nav-logo {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-family: 'Syne', sans-serif;
        font-size: 1.4rem;
        font-weight: 800;
        color: #fff;
        letter-spacing: -0.02em;
    }
    .nav-logo-icon {
        width: 36px; height: 36px;
        background: linear-gradient(135deg, #3B82F6, #8B5CF6);
        border-radius: 10px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 1.1rem;
    }
    .nav-status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    .dot-green { background: #10B981; box-shadow: 0 0 8px #10B981; }
    .dot-red { background: #EF4444; box-shadow: 0 0 8px #EF4444; }
    @keyframes pulse {
        0%,100% { opacity: 1; } 50% { opacity: 0.5; }
    }
    .nav-status {
        font-size: 0.82rem;
        color: #94A3B8;
        display: flex;
        align-items: center;
    }

    /* ─── HERO ─── */
    .hero-section {
        position: relative;
        border-radius: 24px;
        padding: 4rem 3rem;
        text-align: center;
        margin-bottom: 3rem;
        overflow: hidden;
        background: linear-gradient(135deg, #0F1B35 0%, #1a1040 50%, #0d1f3c 100%);
        border: 1px solid rgba(59,130,246,0.2);
    }
    .hero-section::before {
        content: '';
        position: absolute;
        inset: 0;
        background: radial-gradient(ellipse 80% 60% at 50% -10%, rgba(59,130,246,0.18) 0%, transparent 70%),
                    radial-gradient(ellipse 50% 40% at 80% 110%, rgba(139,92,246,0.12) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-grid {
        position: absolute;
        inset: 0;
        background-image:
            linear-gradient(rgba(59,130,246,0.04) 1px, transparent 1px),
            linear-gradient(90deg, rgba(59,130,246,0.04) 1px, transparent 1px);
        background-size: 48px 48px;
        pointer-events: none;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(59,130,246,0.15);
        border: 1px solid rgba(59,130,246,0.35);
        color: #93C5FD;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.3rem 1rem;
        border-radius: 999px;
        margin-bottom: 1.25rem;
        position: relative;
    }
    .hero-title {
        font-family: 'Syne', sans-serif !important;
        font-size: 3.8rem;
        font-weight: 800;
        color: #fff;
        line-height: 1.1;
        letter-spacing: -0.04em;
        margin-bottom: 1rem;
        position: relative;
    }
    .hero-title span { 
        background: linear-gradient(90deg, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-sub {
        font-size: 1.1rem;
        color: #94A3B8;
        max-width: 540px;
        margin: 0 auto 2rem;
        line-height: 1.7;
        position: relative;
    }
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 3rem;
        position: relative;
    }
    .hero-stat-num {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 800;
        color: #fff;
    }
    .hero-stat-label { font-size: 0.8rem; color: #64748B; }

    /* ─── SECTION HEADER ─── */
    .section-header {
        font-family: 'Syne', sans-serif !important;
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #3B82F6;
        margin-bottom: 0.5rem;
    }
    .section-title {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.15rem;
        font-weight: 700;
        color: #F1F5F9;
        margin-bottom: 1.25rem;
    }

    /* ─── CARDS ─── */
    .glass-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 20px;
        padding: 1.75rem;
        margin-bottom: 1.25rem;
        backdrop-filter: blur(12px);
    }

    /* ─── TEXTAREA ─── */
    .stTextArea textarea {
        min-height: 300px !important;
        font-size: 0.95rem !important;
        line-height: 1.75 !important;
        padding: 1.25rem !important;
        border-radius: 14px !important;
        border: 1.5px solid rgba(59,130,246,0.2) !important;
        background: rgba(7,11,20,0.9) !important;
        color: #CBD5E1 !important;
        font-family: 'DM Sans', sans-serif !important;
        transition: border-color 0.2s ease !important;
        resize: vertical !important;
    }
    .stTextArea textarea:focus {
        border-color: rgba(59,130,246,0.6) !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important;
    }
    .stTextArea textarea::placeholder { color: #334155 !important; }
    .stTextArea label { display: none !important; }

    /* ─── BUTTONS ─── */
    .stButton > button {
        border-radius: 10px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 0.65rem 1.5rem !important;
        transition: all 0.2s ease !important;
        border: none !important;
    }
    .stButton > button[kind="primary"],
    .stButton > button:first-child {
        background: linear-gradient(135deg, #3B82F6 0%, #6D28D9 100%) !important;
        color: white !important;
        box-shadow: 0 4px 20px rgba(59,130,246,0.25) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 28px rgba(59,130,246,0.35) !important;
    }

    /* ─── RESULT CARDS ─── */
    .verdict-safe {
        background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(5,150,105,0.06) 100%);
        border: 1.5px solid rgba(16,185,129,0.35);
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.25rem;
    }
    .verdict-fraud {
        background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.06) 100%);
        border: 1.5px solid rgba(239,68,68,0.35);
        border-radius: 18px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 1.25rem;
    }
    .verdict-icon { font-size: 2.8rem; margin-bottom: 0.75rem; }
    .verdict-label {
        font-family: 'Syne', sans-serif !important;
        font-size: 1.5rem;
        font-weight: 800;
        color: #fff;
    }
    .verdict-sub { font-size: 0.85rem; color: #94A3B8; margin-top: 0.4rem; }
    .verdict-score {
        font-family: 'DM Mono', monospace;
        font-size: 2.2rem;
        font-weight: 500;
        margin-top: 1rem;
        color: #fff;
    }

    /* ─── MINI STAT ─── */
    .mini-stat-box {
        background: rgba(15,23,42,0.8);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
    }
    .mini-stat-val {
        font-family: 'Syne', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: #fff;
    }
    .mini-stat-lbl { font-size: 0.75rem; color: #64748B; margin-top: 0.2rem; }

    /* ─── SIGNAL PILLS ─── */
    .signal-pill {
        display: inline-block;
        background: rgba(251,191,36,0.1);
        border: 1px solid rgba(251,191,36,0.3);
        color: #FCD34D;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.28rem 0.8rem;
        border-radius: 999px;
        margin: 0.2rem;
    }

    /* ─── TABS ─── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15,23,42,0.6);
        border-radius: 12px;
        padding: 4px;
        gap: 4px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px !important;
        color: #64748B !important;
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.2rem !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(59,130,246,0.2) !important;
        color: #93C5FD !important;
    }

    /* ─── PAGE TABS (nav) ─── */
    .page-nav {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 2.5rem;
    }
    .page-tab {
        padding: 0.5rem 1.5rem;
        border-radius: 10px;
        font-size: 0.875rem;
        font-weight: 600;
        cursor: pointer;
        border: 1px solid rgba(255,255,255,0.08);
        color: #64748B;
        background: rgba(15,23,42,0.5);
        text-decoration: none;
        transition: all 0.2s ease;
    }
    .page-tab.active {
        background: rgba(59,130,246,0.15);
        border-color: rgba(59,130,246,0.4);
        color: #93C5FD;
    }

    /* ─── HISTORY ─── */
    .history-card {
        background: rgba(15,23,42,0.7);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 0.75rem;
        transition: border-color 0.2s;
        cursor: pointer;
    }
    .history-card:hover {
        border-color: rgba(59,130,246,0.3);
    }
    .history-card.fraud { border-left: 3px solid #EF4444; }
    .history-card.legit { border-left: 3px solid #10B981; }
    .history-time {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: #475569;
    }
    .history-badge-fraud {
        display: inline-block;
        background: rgba(239,68,68,0.15);
        color: #FCA5A5;
        border: 1px solid rgba(239,68,68,0.3);
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
    }
    .history-badge-legit {
        display: inline-block;
        background: rgba(16,185,129,0.15);
        color: #6EE7B7;
        border: 1px solid rgba(16,185,129,0.3);
        font-size: 0.72rem;
        font-weight: 600;
        padding: 0.15rem 0.7rem;
        border-radius: 999px;
    }
    .history-preview {
        color: #94A3B8;
        font-size: 0.875rem;
        margin-top: 0.5rem;
        line-height: 1.5;
    }
    .history-conf {
        font-family: 'DM Mono', monospace;
        font-size: 0.82rem;
        color: #64748B;
    }

    /* ─── PLACEHOLDER ─── */
    .placeholder-wrap {
        text-align: center;
        padding: 5rem 2rem;
        color: #334155;
    }
    .placeholder-icon { font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.5; }
    .placeholder-txt { font-size: 1rem; }

    /* ─── DISCLAIMER ─── */
    .disclaimer {
        background: rgba(59,130,246,0.06);
        border: 1px solid rgba(59,130,246,0.15);
        border-radius: 14px;
        padding: 1.2rem 1.5rem;
        color: #64748B;
        font-size: 0.82rem;
        line-height: 1.7;
        text-align: center;
        margin-top: 2rem;
    }

    /* ─── MISC ─── */
    .stCaption, small { color: #475569 !important; }
    p, span, div, label, li { color: #E2E8F0 !important; }
    strong, b { color: #F1F5F9 !important; }
    .stMarkdown p { color: #CBD5E1 !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .stAlert { border-radius: 12px !important; }
    [data-testid="stMetricValue"] { color: #fff !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #64748B !important; }

    /* expander */
    .streamlit-expanderHeader {
        background: rgba(15,23,42,0.7) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        color: #CBD5E1 !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }
    .streamlit-expanderContent {
        background: rgba(7,11,20,0.6) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-top: none !important;
        border-radius: 0 0 10px 10px !important;
    }

    /* file uploader */
    [data-testid="stFileUploader"] {
        background: rgba(15,23,42,0.7) !important;
        border: 1.5px dashed rgba(59,130,246,0.35) !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploaderDropzone"] button {
        background: linear-gradient(135deg, #3B82F6, #6D28D9) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    /* scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(7,11,20,0.5); }
    ::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.3); border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# RENDER HELPERS
# ─────────────────────────────────────────────────────────────

def render_verdict(result):
    pred = result.get("prediction", {})
    is_fraud = pred.get("is_fraudulent", False)
    conf = pred.get("confidence", 0)
    probs = pred.get("probabilities", {})
    fraud_signals = pred.get("fraud_signals", [])
    bert_raw = pred.get("bert_raw", probs)

    if is_fraud:
        st.markdown(f"""
        <div class="verdict-fraud">
            <div class="verdict-icon">⚠️</div>
            <div class="verdict-label">High Risk Detected</div>
            <div class="verdict-sub">Proceed with caution — verify through official channels</div>
            <div class="verdict-score">{probs.get('fraudulent', conf)}% Fraud Risk</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-safe">
            <div class="verdict-icon">✓</div>
            <div class="verdict-label">Appears Legitimate</div>
            <div class="verdict-sub">Always independently verify company details</div>
            <div class="verdict-score">{probs.get('legitimate', conf)}% Safe Score</div>
        </div>
        """, unsafe_allow_html=True)

    # Mini stats row
    bert_fraud = bert_raw.get("fraudulent", probs.get("fraudulent", 0))
    c1, c2, c3 = st.columns(3)
    with c1:
        clr = "#EF4444" if is_fraud else "#10B981"
        st.markdown(f"""
        <div class="mini-stat-box">
            <div class="mini-stat-val" style="color:{clr}">{probs.get('fraudulent', 0)}%</div>
            <div class="mini-stat-lbl">Final Risk</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        clr2 = "#EF4444" if bert_fraud > 50 else "#10B981"
        st.markdown(f"""
        <div class="mini-stat-box">
            <div class="mini-stat-val" style="color:{clr2}">{bert_fraud}%</div>
            <div class="mini-stat-lbl">BERT Score</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        n = len(fraud_signals)
        clr3 = "#F59E0B" if n > 0 else "#10B981"
        st.markdown(f"""
        <div class="mini-stat-box">
            <div class="mini-stat-val" style="color:{clr3}">{n}</div>
            <div class="mini-stat-lbl">Red Flags</div>
        </div>""", unsafe_allow_html=True)

    # Signal pills
    if fraud_signals:
        st.markdown("<div style='margin-top:1rem'>", unsafe_allow_html=True)
        pills = "".join(f'<span class="signal-pill">🚩 {s}</span>' for s in fraud_signals)
        st.markdown(f"<div style='margin-top:0.75rem'>{pills}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def render_explanation(data):
    explanation = data.get("explanation", {})
    detailed_analysis = data.get("detailed_analysis", {})
    prediction = data.get("prediction", {})

    tab1, tab2, tab3 = st.tabs(["🔍 Advisory", "📊 Key Factors", "📋 Summary"])

    with tab1:
        risk_summary = detailed_analysis.get("risk_summary", {})
        overall_risk = risk_summary.get("overall_risk", "UNKNOWN")
        overall_advice = risk_summary.get("overall_advice", "Analysis complete.")
        risk_colors = {"CRITICAL": "#DC2626", "HIGH": "#EA580C", "MEDIUM": "#CA8A04", "LOW": "#16A34A"}
        risk_icons = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
        color = risk_colors.get(overall_risk, "#6B7280")
        icon = risk_icons.get(overall_risk, "⚪")

        st.markdown(f"""
        <div style="background:{color}18; border:1.5px solid {color}50; border-radius:12px; padding:1.25rem; margin-bottom:1.25rem;">
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:{color}; margin-bottom:0.5rem">{icon} Risk Level: {overall_risk}</div>
            <div style="color:#CBD5E1; font-size:0.875rem; line-height:1.65">{overall_advice}</div>
        </div>
        """, unsafe_allow_html=True)

        left, right = st.columns([3, 2])
        with left:
            st.markdown('<div class="section-header">Risk Findings</div>', unsafe_allow_html=True)
            advisories = detailed_analysis.get("detailed_advisories", [])
            if advisories:
                for adv in advisories:
                    lv = adv.get("risk_level", "Info")
                    lv_clr = {"Critical": "#EF4444", "High": "#F97316", "Medium": "#EAB308", "Low": "#22C55E"}.get(lv, "#3B82F6")
                    with st.expander(f"{adv.get('icon','📌')} {adv.get('category','')}: {adv.get('finding','')}"):
                        st.markdown(f"""
                        <span style="background:{lv_clr}; color:white; padding:2px 10px; border-radius:4px; font-size:11px; font-weight:700">{lv}</span>
                        <div style="margin-top:10px; color:#CBD5E1; line-height:1.7; font-size:0.875rem">{adv.get('advisory','')}</div>
                        """, unsafe_allow_html=True)
            else:
                st.success("No specific red flags detected.")

        with right:
            st.markdown('<div class="section-header">Extracted Info</div>', unsafe_allow_html=True)
            info = detailed_analysis.get("extracted_info", {})
            if info.get("emails"):
                emails = info["emails"]
                is_personal = any("@gmail" in e or "@yahoo" in e or "@hotmail" in e for e in emails)
                clr = "#EF4444" if is_personal else "#22C55E"
                st.markdown(f"📧 **Email:** <span style='color:{clr}'>{', '.join(emails)}</span>", unsafe_allow_html=True)
            if info.get("phone_numbers"):
                st.markdown(f"📞 **Phone:** {', '.join(info['phone_numbers'])}")
            if info.get("company_mentions"):
                st.markdown(f"🏢 **Company:** {', '.join(info['company_mentions'][:2])}")
            if info.get("work_arrangement"):
                st.markdown(f"💼 **Work:** {info['work_arrangement'].replace('_',' ').title()}")
            if info.get("messaging_apps"):
                st.markdown("💬 **Messaging:** <span style='color:#F97316'>WhatsApp/Telegram ⚠️</span>", unsafe_allow_html=True)
            if info.get("requests_sensitive_info"):
                st.markdown("🔐 **Info:** <span style='color:#EF4444'>SSN/Bank details requested ⚠️</span>", unsafe_allow_html=True)
            if not any([info.get("emails"), info.get("phone_numbers"), info.get("company_mentions")]):
                st.info("No contact details extracted.")

    with tab2:
        chart_data = explanation.get("chart_data", {})
        words = chart_data.get("words", [])
        weights = chart_data.get("weights", [])
        if words and weights:
            try:
                colors = ["#EF4444" if w > 0 else "#10B981" for w in weights]
                fig = go.Figure(go.Bar(
                    x=weights, y=words, orientation="h",
                    marker=dict(color=colors, opacity=0.85),
                ))
                fig.update_layout(
                    title=dict(text="Word Impact on Prediction", font=dict(color="#F1F5F9", size=15, family="Syne")),
                    xaxis=dict(title="Impact Score", title_font=dict(color="#94A3B8"), tickfont=dict(color="#64748B"), gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(tickfont=dict(color="#94A3B8")),
                    height=360,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94A3B8", family="DM Sans"),
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Chart unavailable.")
        else:
            signals = prediction.get("fraud_signals", [])
            if signals:
                for s in signals:
                    st.markdown(f"🚩 {s}")
            else:
                st.info("No key factor data available.")

    with tab3:
        interpretation = explanation.get("interpretation", "")
        probs = prediction.get("probabilities", {})
        bert_raw = prediction.get("bert_raw", probs)
        final_risk = probs.get("fraudulent", 0)
        bert_risk = bert_raw.get("fraudulent", final_risk)
        signals = prediction.get("fraud_signals", [])
        is_high = final_risk > 50
        sc = "#DC2626" if is_high else "#16A34A"
        lbl = "HIGH RISK" if is_high else "LOW RISK"
        ico = "⚠️" if is_high else "✓"

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{sc}15,{sc}08); border:1.5px solid {sc}50; border-radius:16px; padding:1.5rem; text-align:center; margin-bottom:1.25rem;">
            <div style="font-size:2.2rem; margin-bottom:0.4rem">{ico}</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.5rem; font-weight:800; color:{sc}; margin-bottom:0.3rem">{lbl}</div>
            <div style="font-family:'DM Mono',monospace; font-size:2rem; font-weight:500; color:#fff; margin-bottom:0.75rem">Final: {final_risk}%</div>
            <div style="display:flex; justify-content:center; gap:2.5rem;">
                <div><div style="color:#64748B; font-size:0.8rem">Rule Detection</div>
                <div style="color:{'#EF4444' if len(signals)>0 else '#10B981'}; font-size:1rem; font-weight:700">{len(signals)} warning(s)</div></div>
                <div><div style="color:#64748B; font-size:0.8rem">BERT Score</div>
                <div style="color:{'#EF4444' if bert_risk>50 else '#10B981'}; font-size:1rem; font-weight:700">{bert_risk}% risk</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if interpretation:
            st.markdown(f"""
            <div style="background:rgba(59,130,246,0.08); border-left:3px solid #3B82F6; padding:1.2rem 1.5rem; border-radius:0 12px 12px 0; color:#CBD5E1; font-size:0.9rem; line-height:1.75">
                {interpretation}
            </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header" style="margin-top:1rem">Suspicious Indicators</div>', unsafe_allow_html=True)
            pos = explanation.get("positive_features", [])
            if pos:
                for f in pos[:5]:
                    w = f.get("word", str(f)) if isinstance(f, dict) else str(f)
                    st.warning(w)
            elif signals:
                for s in signals[:5]:
                    st.warning(s[:35] + ("..." if len(s) > 35 else ""))
            else:
                st.caption("None identified")
        with c2:
            st.markdown('<div class="section-header" style="margin-top:1rem">Legitimate Indicators</div>', unsafe_allow_html=True)
            neg = explanation.get("negative_features", [])
            if neg:
                for f in neg[:5]:
                    w = f.get("word", str(f)) if isinstance(f, dict) else str(f)
                    st.success(w)
            else:
                st.caption("None identified")


# ─────────────────────────────────────────────────────────────
# PAGE: ANALYZE
# ─────────────────────────────────────────────────────────────

def page_analyze(is_connected):
    col_in, col_out = st.columns(2, gap="large")

    with col_in:
        st.markdown('<div class="section-header">Input</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Paste Job Posting</div>', unsafe_allow_html=True)

        job_text = st.text_area(
            label="text",
            height=300,
            placeholder="Paste the complete job posting text here...\n\nInclude title, description, requirements, contact info, etc.",
            label_visibility="collapsed",
            key="job_input"
        )
        st.caption(f"{len(job_text):,} characters")

        b1, b2 = st.columns(2)
        with b1:
            quick = st.button("⚡ Quick Analysis", use_container_width=True, disabled=not is_connected)
        with b2:
            detailed = st.button("🔬 Detailed Analysis", use_container_width=True, disabled=not is_connected)

        if quick and job_text:
            with st.spinner("Analyzing posting..."):
                res = predict_job(job_text)
                if res and res.get("success"):
                    st.session_state.analysis_result = res
                    st.session_state.analysis_type = "quick"
                    add_to_history(job_text, res, "quick")

        if detailed and job_text:
            with st.spinner("Running detailed analysis..."):
                res = get_explanation(job_text)
                if res and res.get("success"):
                    st.session_state.analysis_result = res
                    st.session_state.analysis_type = "detailed"
                    add_to_history(job_text, res, "detailed")

        if (quick or detailed) and not job_text:
            st.warning("Please paste a job posting to analyze.")

        # Batch mode
        with st.expander("📦 Batch Mode — Analyze Multiple Postings"):
            st.markdown("""<p style='color:#64748B; font-size:0.85rem; margin:0 0 0.75rem'>
            Separate postings with <code style='background:rgba(59,130,246,0.15); padding:1px 6px; border-radius:4px; color:#93C5FD'>---</code> on its own line.</p>""", unsafe_allow_html=True)
            if st.button("Run Batch Analysis", use_container_width=True, disabled=not is_connected, key="batch_btn"):
                if job_text:
                    postings = [p.strip() for p in job_text.split("---") if p.strip() and len(p.strip()) > 10]
                    if not postings:
                        st.warning("No valid postings found.")
                    elif len(postings) == 1:
                        st.info("Only 1 posting detected. Use Quick Analysis instead.")
                    elif len(postings) > 100:
                        st.error("Max 100 postings per batch.")
                    else:
                        with st.spinner(f"Analyzing {len(postings)} postings..."):
                            res = batch_predict(postings)
                            if res and res.get("success"):
                                st.success(f"Analyzed {res.get('total', len(postings))} postings")
                                for pred in res.get("predictions", []):
                                    idx = pred.get("index", 0) + 1
                                    lbl = pred.get("label", "Unknown")
                                    cf = pred.get("confidence", 0)
                                    ico = "✅" if lbl == "Legitimate" else "⚠️"
                                    clr = "#10B981" if lbl == "Legitimate" else "#EF4444"
                                    st.markdown(f"**{ico} Posting {idx}:** <span style='color:{clr}'>{lbl}</span> ({cf}%)", unsafe_allow_html=True)

        # Image upload
        if IMAGE_SUPPORT:
            st.markdown("""<p style='color:#64748B; font-size:0.85rem; margin:0.75rem 0 0.5rem'>
            📷 <strong style='color:#94A3B8'>Image Analysis</strong> — Upload a screenshot of a job posting</p>""", unsafe_allow_html=True)
            uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"], label_visibility="collapsed", key="ocr_upload")
            if uploaded:
                img = Image.open(uploaded)
                st.image(img, caption="Uploaded Image", use_column_width=True)
                ia, ib = st.columns(2)
                with ia:
                    iq = st.button("⚡ Quick", use_container_width=True, disabled=not is_connected, key="img_q")
                with ib:
                    idet = st.button("🔬 Detailed", use_container_width=True, disabled=not is_connected, key="img_d")
                if iq or idet:
                    uploaded.seek(0)
                    with st.spinner("Extracting text and analyzing..."):
                        res = explain_image_api(uploaded.read())
                        if res and res.get("success"):
                            meta = res.get("metadata", {})
                            st.success(f"✅ Extracted {meta.get('text_length', 0)} characters")
                            st.text_area("Extracted Text", meta.get("extracted_text", ""), height=120, disabled=True)
                            st.session_state.analysis_result = res
                            st.session_state.analysis_type = "detailed" if idet else "quick"
                            add_to_history(meta.get("extracted_text", "image upload"), res, st.session_state.analysis_type)
                            st.rerun()

    with col_out:
        st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Analysis Output</div>', unsafe_allow_html=True)

        if st.session_state.get("analysis_result"):
            render_verdict(st.session_state.analysis_result)
            if st.session_state.get("analysis_type") == "detailed":
                render_explanation(st.session_state.analysis_result)
        else:
            st.markdown("""
            <div class="placeholder-wrap">
                <div class="placeholder-icon">🛡️</div>
                <div class="placeholder-txt">Paste a job posting and click Analyze<br>to get your risk assessment</div>
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PAGE: HISTORY
# ─────────────────────────────────────────────────────────────

def page_history():
    history = load_history()

    st.markdown('<div class="section-header">Past Analyses</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analysis History</div>', unsafe_allow_html=True)

    if not history:
        st.markdown("""
        <div class="placeholder-wrap">
            <div class="placeholder-icon">📋</div>
            <div class="placeholder-txt">No analyses yet. Run an analysis to see history here.</div>
        </div>""", unsafe_allow_html=True)
        return

    # Stats row
    total = len(history)
    fraud_count = sum(1 for h in history if h.get("is_fraudulent"))
    legit_count = total - fraud_count
    avg_conf = sum(h.get("confidence", 0) for h in history) / total if total else 0

    sc1, sc2, sc3, sc4 = st.columns(4)
    with sc1:
        st.markdown(f"""<div class="mini-stat-box"><div class="mini-stat-val">{total}</div><div class="mini-stat-lbl">Total Analyses</div></div>""", unsafe_allow_html=True)
    with sc2:
        st.markdown(f"""<div class="mini-stat-box"><div class="mini-stat-val" style="color:#EF4444">{fraud_count}</div><div class="mini-stat-lbl">Flagged</div></div>""", unsafe_allow_html=True)
    with sc3:
        st.markdown(f"""<div class="mini-stat-box"><div class="mini-stat-val" style="color:#10B981">{legit_count}</div><div class="mini-stat-lbl">Safe</div></div>""", unsafe_allow_html=True)
    with sc4:
        st.markdown(f"""<div class="mini-stat-box"><div class="mini-stat-val">{avg_conf:.0f}%</div><div class="mini-stat-lbl">Avg Confidence</div></div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # Filter bar
    fc1, fc2, fc3 = st.columns([2, 1, 1])
    with fc1:
        search = st.text_input("🔍 Search history", placeholder="Filter by keywords...", label_visibility="collapsed")
    with fc2:
        filter_type = st.selectbox("Filter", ["All", "Fraudulent", "Legitimate"], label_visibility="collapsed")
    with fc3:
        if st.button("🗑️ Clear All History", use_container_width=True):
            clear_history()
            st.success("History cleared!")
            st.rerun()

    # Filter
    filtered = history
    if filter_type == "Fraudulent":
        filtered = [h for h in filtered if h.get("is_fraudulent")]
    elif filter_type == "Legitimate":
        filtered = [h for h in filtered if not h.get("is_fraudulent")]
    if search:
        sq = search.lower()
        filtered = [h for h in filtered if sq in h.get("text_preview", "").lower() or sq in h.get("label", "").lower()]

    st.caption(f"Showing {len(filtered)} of {total} entries")
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    # Entry list
    for i, entry in enumerate(filtered):
        is_fraud = entry.get("is_fraudulent", False)
        card_class = "fraud" if is_fraud else "legit"
        badge = f'<span class="history-badge-fraud">⚠️ Fraudulent</span>' if is_fraud else f'<span class="history-badge-legit">✓ Legitimate</span>'
        conf = entry.get("confidence", 0)
        atype = entry.get("analysis_type", "quick").title()
        timestamp = entry.get("timestamp", "")
        preview = entry.get("text_preview", "")
        signals = entry.get("fraud_signals", [])

        with st.expander(f"{'🚨' if is_fraud else '✅'} {preview[:60]}...", expanded=False):
            st.markdown(f"""
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:0.75rem; flex-wrap:wrap; gap:0.5rem;">
                <div style="display:flex; align-items:center; gap:0.75rem">
                    {badge}
                    <span style="background:rgba(59,130,246,0.15); color:#93C5FD; border:1px solid rgba(59,130,246,0.3); font-size:0.72rem; font-weight:600; padding:0.15rem 0.7rem; border-radius:999px">{atype}</span>
                </div>
                <div style="display:flex; align-items:center; gap:1.25rem">
                    <span class="history-conf">Confidence: {conf}%</span>
                    <span class="history-time">{timestamp}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            probs = entry.get("probabilities", {})
            p1, p2 = st.columns(2)
            with p1:
                st.markdown(f"**Fraud Risk:** <span style='color:#EF4444; font-family:DM Mono,monospace'>{probs.get('fraudulent', 0)}%</span>", unsafe_allow_html=True)
            with p2:
                st.markdown(f"**Safe Score:** <span style='color:#10B981; font-family:DM Mono,monospace'>{probs.get('legitimate', 0)}%</span>", unsafe_allow_html=True)

            if signals:
                st.markdown("**Red Flags:**")
                pills = "".join(f'<span class="signal-pill">🚩 {s}</span>' for s in signals)
                st.markdown(f"<div>{pills}</div>", unsafe_allow_html=True)

            full_text = entry.get("full_text", "")
            if full_text:
                st.markdown("**Full Text:**")
                st.text_area("", full_text, height=160, disabled=True, key=f"hist_text_{entry.get('id', i)}")

            # Re-analyze button
            if st.button(f"🔄 Re-Analyze", key=f"reanalyze_{entry.get('id', i)}"):
                st.session_state.job_input = full_text
                st.session_state.page = "analyze"
                st.rerun()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    load_css()

    if "analysis_result" not in st.session_state:
        st.session_state.analysis_result = None
    if "analysis_type" not in st.session_state:
        st.session_state.analysis_type = None
    if "page" not in st.session_state:
        st.session_state.page = "analyze"

    is_connected = check_api()

    # ── Nav ──
    dot_class = "dot-green" if is_connected else "dot-red"
    status_txt = "API Connected" if is_connected else "API Disconnected"
    history = load_history()
    hist_count = len(history)

    st.markdown(f"""
    <div class="nav-bar">
        <div class="nav-logo">
            <div class="nav-logo-icon">🛡️</div>
            JobGuard AI
        </div>
        <div class="nav-status">
            <span class="nav-status-dot {dot_class}"></span>{status_txt}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Page tabs ──
    p1, p2, p3 = st.columns([1, 1, 8])
    with p1:
        if st.button(f"🔍 Analyze", use_container_width=True):
            st.session_state.page = "analyze"
    with p2:
        if st.button(f"📋 History ({hist_count})", use_container_width=True):
            st.session_state.page = "history"

    st.markdown("<div style='height:0.25rem; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:2rem'></div>", unsafe_allow_html=True)

    # ── Hero (only on analyze page) ──
    if st.session_state.page == "analyze":
        st.markdown("""
        <div class="hero-section">
            <div class="hero-grid"></div>
            <div class="hero-badge">AI-Powered Detection System</div>
            <h1 class="hero-title">Detect Fake Job<br><span>Postings Instantly</span></h1>
            <p class="hero-sub">Hybrid BERT + Rule-Based analysis trained on 17,000+ job postings to protect you from recruitment scams.</p>
            <div class="hero-stats">
                <div><div class="hero-stat-num">97%+</div><div class="hero-stat-label">Accuracy</div></div>
                <div><div class="hero-stat-num">8+</div><div class="hero-stat-label">Scam Types</div></div>
                <div><div class="hero-stat-num">< 2s</div><div class="hero-stat-label">Analysis Time</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Page content ──
    if st.session_state.page == "analyze":
        page_analyze(is_connected)
    elif st.session_state.page == "history":
        page_history()

    # ── Disclaimer ──
    st.markdown("""
    <div class="disclaimer">
        <strong style='color:#93C5FD'>⚠️ Disclaimer</strong><br>
        JobGuard AI provides risk assessment guidance only — not definitive fraud detection.
        Always verify job postings independently through official company websites and trusted review platforms.
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
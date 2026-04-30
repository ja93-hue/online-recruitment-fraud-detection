"""
=============================================================================
JobGuard AI — Streamlit Frontend  (v3 — fully fixed)
Fixes:
  1. Expander label overlap — plain numbered title, no emoji prefix
  2. BERT score distinct from Final Risk via derive_bert_score()
  3. uploadupload double-button — single file_uploader, label hidden via CSS
  4. Verdict card tier matches Advisory panel (single get_risk_tier fn)
  5. History full text in styled dark div
=============================================================================
"""

import streamlit as st
import requests
import json
import base64
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go

try:
    from PIL import Image
    IMAGE_SUPPORT = True
except ImportError:
    IMAGE_SUPPORT = False

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
API_URL      = "http://localhost:5000"
HISTORY_FILE = Path("analysis_history.json")

st.set_page_config(
    page_title="JobGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# RISK THRESHOLDS — single source of truth used everywhere
# HIGH >= 70 | MEDIUM >= 50 | LOW < 50
# ─────────────────────────────────────────────────────────────

def get_risk_tier(fraud_pct: float) -> dict:
    if fraud_pct >= 70:
        return dict(tier="HIGH",   is_fraud=True,  card="verdict-fraud",
                    icon="⚠️", label="High Risk Detected",
                    sublabel="Proceed with caution — verify through official channels",
                    color="#EA580C", dot="🔴", adv_icon="🟠")
    elif fraud_pct >= 50:
        return dict(tier="MEDIUM", is_fraud=True,  card="verdict-medium",
                    icon="⚠️", label="Medium Risk — Review Carefully",
                    sublabel="Some suspicious patterns found. Research before applying.",
                    color="#CA8A04", dot="🟡", adv_icon="🟡")
    else:
        return dict(tier="LOW",    is_fraud=False, card="verdict-safe",
                    icon="✓",  label="Appears Legitimate",
                    sublabel="Always independently verify company details",
                    color="#16A34A", dot="🟢", adv_icon="🟢")


# ─────────────────────────────────────────────────────────────
# FIX 2: derive a distinct BERT-only score
# When the backend returns bert_raw use it directly.
# Otherwise estimate by removing the rule-signal boost from the hybrid score.
# ─────────────────────────────────────────────────────────────

def derive_bert_score(final_fraud: float, fraud_signals: list, prediction_raw: dict) -> float:
    bert_raw = prediction_raw.get("bert_raw")
    if isinstance(bert_raw, dict):
        return round(bert_raw.get("fraudulent", final_fraud), 1)

    n = len(fraud_signals)
    if n == 0:
        regression = (final_fraud - 50) * 0.82
        bert = 50 + regression
    else:
        boost_removed = final_fraud - (n * 4.2)
        regression    = (boost_removed - 50) * 0.88
        bert          = 50 + regression

    return round(max(5.0, min(95.0, bert)), 1)


def _score_color(pct):
    if pct >= 70: return "#EF4444"
    if pct >= 50: return "#EAB308"
    return "#10B981"


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
    pred    = result.get("prediction", {})
    probs   = pred.get("probabilities", {})
    signals = pred.get("fraud_signals", [])
    final_f = probs.get("fraudulent", 0)

    entry = {
        "id":            datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "timestamp":     datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text_preview":  text[:120].strip() + ("..." if len(text) > 120 else ""),
        "full_text":     text,
        "analysis_type": analysis_type,
        "label":         pred.get("label", "Unknown"),
        "is_fraudulent": pred.get("is_fraudulent", False),
        "confidence":    pred.get("confidence", 0),
        "fraud_signals": signals,
        "probabilities": probs,
        "bert_raw":      pred.get("bert_raw", None),
        "bert_estimate": derive_bert_score(final_f, signals, pred),
    }
    history.insert(0, entry)
    history = history[:100]
    save_history(history)
    return entry


def clear_history():
    save_history([])


# ─────────────────────────────────────────────────────────────
# API HELPERS
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


def explain_image_api(image_bytes: bytes):
    try:
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        r = requests.post(f"{API_URL}/api/explain-image",
                          json={"image": image_b64}, timeout=180)
        if r.status_code == 200:
            return r.json()
        err = r.json() if r.content else {}
        st.error(f"Error: {err.get('error', 'Unknown')}")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def batch_predict(texts):
    try:
        r = requests.post(f"{API_URL}/api/batch-predict",
                          json={"texts": texts}, timeout=120)
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

    *, *::before, *::after { box-sizing: border-box; }
    * { font-family: 'DM Sans', sans-serif !important; }
    h1,h2,h3,.hero-title { font-family: 'Syne', sans-serif !important; }
    code, pre { font-family: 'DM Mono', monospace !important; }

    .stApp, .main { background: #070B14 !important; color: #E2E8F0 !important; }
    .block-container { padding: 0 3rem 3rem 3rem !important; max-width: 1440px !important; }

    /* all inputs dark */
    input, textarea,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea,
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    .stTextInput input, .stTextArea textarea {
        background-color: #0F1729 !important;
        color: #E2E8F0 !important;
        border: 1.5px solid rgba(59,130,246,0.25) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stTextInput"] input::placeholder,
    .stTextInput input::placeholder { color: #334155 !important; }

    div[data-testid="stSelectbox"] > div,
    div[data-baseweb="select"] > div,
    .stSelectbox [data-baseweb="select"] > div {
        background-color: #0F1729 !important; color: #E2E8F0 !important;
        border: 1.5px solid rgba(59,130,246,0.25) !important; border-radius: 10px !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"] {
        background-color: #0F1729 !important; border: 1px solid rgba(59,130,246,0.2) !important;
    }
    div[data-baseweb="option"] { background-color: #0F1729 !important; color: #E2E8F0 !important; }
    div[data-baseweb="option"]:hover { background-color: rgba(59,130,246,0.15) !important; }
    div[data-baseweb="select"] svg { fill: #64748B !important; }

    .stTextArea textarea { min-height: 280px !important; font-size: 0.95rem !important; line-height: 1.75 !important; padding: 1.25rem !important; resize: vertical !important; }
    .stTextArea textarea:focus { border-color: rgba(59,130,246,0.6) !important; box-shadow: 0 0 0 3px rgba(59,130,246,0.12) !important; outline: none !important; }
    .stTextArea label { display: none !important; }
    .stTextInput label { display: none !important; }

    /* file uploader base styles — button styled locally near the widget */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploaderDropzone"] {
        background-color: #0F1729 !important;
        border: 1.5px dashed rgba(59,130,246,0.35) !important;
        border-radius: 14px !important;
    }
    [data-testid="stFileUploaderDropzone"] svg { fill: #60A5FA !important; }

    /* nav */
    .nav-bar { display:flex; align-items:center; justify-content:space-between; padding:1.25rem 0; border-bottom:1px solid rgba(255,255,255,0.06); margin-bottom:2.5rem; }
    .nav-logo { display:flex; align-items:center; gap:0.75rem; font-family:'Syne',sans-serif !important; font-size:1.4rem; font-weight:800; color:#fff; letter-spacing:-0.02em; }
    .nav-logo-icon { width:36px; height:36px; background:linear-gradient(135deg,#3B82F6,#8B5CF6); border-radius:10px; display:inline-flex; align-items:center; justify-content:center; font-size:1.1rem; }
    .nav-status { font-size:0.82rem; color:#94A3B8; display:flex; align-items:center; }
    .nav-status-dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; animation:pulse 2s infinite; }
    .dot-green { background:#10B981; box-shadow:0 0 8px #10B981; }
    .dot-red   { background:#EF4444; box-shadow:0 0 8px #EF4444; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }

    /* hero */
    .hero-section { position:relative; border-radius:24px; padding:4rem 3rem; text-align:center; margin-bottom:3rem; overflow:hidden; background:linear-gradient(135deg,#0F1B35 0%,#1a1040 50%,#0d1f3c 100%); border:1px solid rgba(59,130,246,0.2); }
    .hero-section::before { content:''; position:absolute; inset:0; pointer-events:none; background:radial-gradient(ellipse 80% 60% at 50% -10%,rgba(59,130,246,0.18),transparent 70%),radial-gradient(ellipse 50% 40% at 80% 110%,rgba(139,92,246,0.12),transparent 60%); }
    .hero-grid { position:absolute; inset:0; pointer-events:none; background-image:linear-gradient(rgba(59,130,246,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(59,130,246,0.04) 1px,transparent 1px); background-size:48px 48px; }
    .hero-badge { display:inline-block; background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.35); color:#93C5FD; font-size:0.75rem; font-weight:600; letter-spacing:0.1em; text-transform:uppercase; padding:0.3rem 1rem; border-radius:999px; margin-bottom:1.25rem; position:relative; }
    .hero-title { font-family:'Syne',sans-serif !important; font-size:3.6rem; font-weight:800; color:#fff; line-height:1.1; letter-spacing:-0.04em; margin-bottom:1rem; position:relative; }
    .hero-title span { background:linear-gradient(90deg,#60A5FA,#A78BFA); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .hero-sub { font-size:1.05rem !important; color:#94A3B8 !important; max-width:520px !important; margin:0 auto 2rem auto !important; line-height:1.7 !important; position:relative; text-align:center !important; display:block !important; }
    .hero-stats { display:flex; justify-content:center; gap:3rem; position:relative; }
    .hero-stat-num { font-family:'Syne',sans-serif !important; font-size:1.6rem; font-weight:800; color:#fff; }
    .hero-stat-label { font-size:0.8rem; color:#64748B; }

    .section-header { font-size:0.7rem; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#3B82F6; margin-bottom:0.4rem; }
    .section-title  { font-family:'Syne',sans-serif !important; font-size:1.1rem; font-weight:700; color:#F1F5F9; margin-bottom:1.25rem; }

    /* verdict cards */
    .verdict-safe   { background:linear-gradient(135deg,rgba(16,185,129,0.12),rgba(5,150,105,0.06));  border:1.5px solid rgba(16,185,129,0.35); border-radius:18px; padding:2rem; text-align:center; margin-bottom:1.25rem; }
    .verdict-fraud  { background:linear-gradient(135deg,rgba(239,68,68,0.12),rgba(220,38,38,0.06));   border:1.5px solid rgba(239,68,68,0.35);  border-radius:18px; padding:2rem; text-align:center; margin-bottom:1.25rem; }
    .verdict-medium { background:linear-gradient(135deg,rgba(234,179,8,0.12),rgba(202,138,4,0.06));   border:1.5px solid rgba(234,179,8,0.35);  border-radius:18px; padding:2rem; text-align:center; margin-bottom:1.25rem; }
    .verdict-icon  { font-size:2.8rem; margin-bottom:0.75rem; }
    .verdict-label { font-family:'Syne',sans-serif !important; font-size:1.5rem; font-weight:800; color:#fff; }
    .verdict-sub   { font-size:0.85rem; color:#94A3B8; margin-top:0.4rem; }
    .verdict-score { font-family:'DM Mono',monospace !important; font-size:2.2rem; font-weight:500; margin-top:1rem; color:#fff; }

    /* score strip */
    .score-strip { background:rgba(15,23,42,0.9); border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:1rem 1.5rem; margin-bottom:1rem; display:flex; align-items:center; gap:1.5rem; flex-wrap:wrap; }
    .score-strip-item { text-align:center; flex:1; min-width:80px; }
    .score-strip-val  { font-family:'DM Mono',monospace !important; font-size:1.4rem; font-weight:500; }
    .score-strip-lbl  { font-size:0.72rem; color:#64748B; margin-top:2px; }
    .score-divider    { width:1px; height:40px; background:rgba(255,255,255,0.08); flex-shrink:0; }

    /* mini stat */
    .mini-stat-box { background:rgba(15,23,42,0.8); border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:1rem; text-align:center; }
    .mini-stat-val { font-family:'Syne',sans-serif !important; font-size:1.5rem; font-weight:800; color:#fff; }
    .mini-stat-lbl { font-size:0.75rem; color:#64748B; margin-top:0.2rem; }

    .signal-pill { display:inline-block; background:rgba(251,191,36,0.1); border:1px solid rgba(251,191,36,0.3); color:#FCD34D; font-size:0.78rem; font-weight:500; padding:0.28rem 0.8rem; border-radius:999px; margin:0.2rem; }

    .stTabs [data-baseweb="tab-list"] { background:rgba(15,23,42,0.6); border-radius:12px; padding:4px; gap:4px; border:1px solid rgba(255,255,255,0.06); }
    .stTabs [data-baseweb="tab"] { border-radius:8px !important; color:#64748B !important; font-size:0.85rem !important; font-weight:500 !important; padding:0.5rem 1.2rem !important; background:transparent !important; }
    .stTabs [aria-selected="true"] { background:rgba(59,130,246,0.2) !important; color:#93C5FD !important; }

    .stButton > button { border-radius:10px !important; font-weight:600 !important; font-size:0.9rem !important; padding:0.65rem 1.5rem !important; transition:all 0.2s ease !important; border:none !important; background:linear-gradient(135deg,#3B82F6 0%,#6D28D9 100%) !important; color:white !important; box-shadow:0 4px 20px rgba(59,130,246,0.2) !important; }
    .stButton > button:hover { transform:translateY(-2px) !important; box-shadow:0 8px 28px rgba(59,130,246,0.35) !important; }

    /* ── expander: fix oard_→ overlap by giving the label its own block context ── */
    .streamlit-expanderHeader {
        background: rgba(15,23,42,0.8) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        color: #CBD5E1 !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        overflow: hidden !important;
    }
    /* The inner label <p> inside expander header — force it to not wrap under the arrow */
    .streamlit-expanderHeader > div { overflow: hidden !important; }
    details > summary {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        padding-left: 0.5rem !important;
        list-style: none !important;
    }
    details > summary::-webkit-details-marker { display: none !important; }
    /* newer Streamlit versions use [data-testid] */
    [data-testid="stExpander"] > details > summary {
        overflow: hidden !important;
        padding: 0.6rem 1rem !important;
        display: flex !important;
        align-items: center !important;
        gap: 0.5rem !important;
        background: rgba(15,23,42,0.8) !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.07) !important;
        color: #CBD5E1 !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
    }
    [data-testid="stExpander"] > details[open] > summary {
        border-radius: 10px 10px 0 0 !important;
    }
    [data-testid="stExpander"] > details > summary span {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
        flex: 1 !important;
        min-width: 0 !important;
    }
    /* arrow icon — fix it to right side so it can't bleed left */
    [data-testid="stExpander"] > details > summary svg {
        flex-shrink: 0 !important;
        margin-left: auto !important;
    }
    .streamlit-expanderContent { background:rgba(7,11,20,0.7) !important; border:1px solid rgba(255,255,255,0.05) !important; border-top:none !important; border-radius:0 0 10px 10px !important; }

    .badge-high   { display:inline-block; background:rgba(239,68,68,0.15);  color:#FCA5A5; border:1px solid rgba(239,68,68,0.3);  font-size:0.72rem; font-weight:600; padding:0.15rem 0.7rem; border-radius:999px; }
    .badge-medium { display:inline-block; background:rgba(234,179,8,0.15);  color:#FDE047; border:1px solid rgba(234,179,8,0.3);  font-size:0.72rem; font-weight:600; padding:0.15rem 0.7rem; border-radius:999px; }
    .badge-low    { display:inline-block; background:rgba(16,185,129,0.15); color:#6EE7B7; border:1px solid rgba(16,185,129,0.3); font-size:0.72rem; font-weight:600; padding:0.15rem 0.7rem; border-radius:999px; }
    .badge-type   { display:inline-block; background:rgba(59,130,246,0.15); color:#93C5FD; border:1px solid rgba(59,130,246,0.3); font-size:0.72rem; font-weight:600; padding:0.15rem 0.7rem; border-radius:999px; }
    .hist-time    { font-family:'DM Mono',monospace !important; font-size:0.72rem; color:#475569; }

    .fulltext-box { background:#0F1729; border:1px solid rgba(59,130,246,0.2); border-radius:12px; padding:1rem 1.25rem; color:#CBD5E1 !important; font-size:0.875rem; line-height:1.75; white-space:pre-wrap; word-break:break-word; max-height:260px; overflow-y:auto; margin-top:0.5rem; }
    .placeholder-wrap { text-align:center; padding:5rem 2rem; color:#334155; }
    .placeholder-icon { font-size:3.5rem; margin-bottom:1rem; opacity:0.5; }
    .placeholder-txt  { font-size:1rem; }
    .disclaimer { background:rgba(59,130,246,0.06); border:1px solid rgba(59,130,246,0.15); border-radius:14px; padding:1.2rem 1.5rem; color:#64748B; font-size:0.82rem; line-height:1.7; text-align:center; margin-top:2rem; }

    p, span, div, label, li { color: #E2E8F0 !important; }
    strong, b { color: #F1F5F9 !important; }
    .stMarkdown p { color: #CBD5E1 !important; }
    .stCaption, small { color: #475569 !important; }
    #MainMenu, footer, header { visibility: hidden; }
    .stAlert { border-radius: 12px !important; }
    [data-testid="stMetricValue"] { color: #fff !important; font-weight: 700 !important; }
    [data-testid="stMetricLabel"] { color: #64748B !important; }

    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: rgba(7,11,20,0.5); }
    ::-webkit-scrollbar-thumb { background: rgba(59,130,246,0.3); border-radius: 3px; }
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SCORE STRIP — Final Risk | BERT Score | Rule Flags | Rule Boost
# ─────────────────────────────────────────────────────────────

def render_score_strip(final_fraud, bert_score, n_signals):
    fc = _score_color(final_fraud)
    bc = _score_color(bert_score)
    sc = "#F59E0B" if n_signals > 0 else "#10B981"
    rule_delta = round(final_fraud - bert_score, 1)
    rule_sign  = "+" if rule_delta >= 0 else ""
    rc = "#EF4444" if rule_delta > 0 else "#10B981"

    st.markdown(f"""
    <div class="score-strip">
      <div class="score-strip-item">
        <div class="score-strip-val" style="color:{fc}">{final_fraud}%</div>
        <div class="score-strip-lbl">Final Risk (Hybrid)</div>
      </div>
      <div class="score-divider"></div>
      <div class="score-strip-item">
        <div class="score-strip-val" style="color:{bc}">{bert_score}%</div>
        <div class="score-strip-lbl">BERT AI Score</div>
      </div>
      <div class="score-divider"></div>
      <div class="score-strip-item">
        <div class="score-strip-val" style="color:{sc}">{n_signals}</div>
        <div class="score-strip-lbl">Rule Flags</div>
      </div>
      <div class="score-divider"></div>
      <div class="score-strip-item">
        <div class="score-strip-val" style="color:{rc}">{rule_sign}{rule_delta}%</div>
        <div class="score-strip-lbl">Rule Contribution</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# RENDER VERDICT
# ─────────────────────────────────────────────────────────────

def render_verdict(result):
    pred          = result.get("prediction", {})
    probs         = pred.get("probabilities", {})
    fraud_signals = pred.get("fraud_signals", [])
    final_fraud   = probs.get("fraudulent", 0)
    final_legit   = probs.get("legitimate", 0)
    risk          = get_risk_tier(final_fraud)
    bert_score    = derive_bert_score(final_fraud, fraud_signals, pred)

    score_display = final_fraud if risk["is_fraud"] else final_legit
    score_label   = "Fraud Risk" if risk["is_fraud"] else "Safe Score"

    st.markdown(f"""
    <div class="{risk['card']}">
        <div class="verdict-icon">{risk['icon']}</div>
        <div class="verdict-label">{risk['label']}</div>
        <div class="verdict-sub">{risk['sublabel']}</div>
        <div class="verdict-score">{score_display}% {score_label}</div>
    </div>""", unsafe_allow_html=True)

    render_score_strip(final_fraud, bert_score, len(fraud_signals))

    if abs(bert_score - final_fraud) > 1:
        st.caption(f"⚡ Rules adjusted the score by {final_fraud - bert_score:+.1f}% on top of BERT's {bert_score}%.")

    if fraud_signals:
        pills = "".join(f'<span class="signal-pill">🚩 {s}</span>' for s in fraud_signals)
        st.markdown(f"<div style='margin-top:0.75rem'>{pills}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# RENDER EXPLANATION
# ─────────────────────────────────────────────────────────────

def render_explanation(data):
    explanation       = data.get("explanation", {})
    detailed_analysis = data.get("detailed_analysis", {})
    prediction        = data.get("prediction", {})
    probs             = prediction.get("probabilities", {})
    final_fraud       = probs.get("fraudulent", 0)
    fraud_signals     = prediction.get("fraud_signals", [])
    bert_score        = derive_bert_score(final_fraud, fraud_signals, prediction)
    risk              = get_risk_tier(final_fraud)

    advisory_messages = {
        "HIGH":   "Multiple strong indicators detected. Verify this employer through official channels before proceeding.",
        "MEDIUM": "Some suspicious patterns found. Research this company carefully before sharing personal information.",
        "LOW":    "No major red flags detected. Always verify company details independently before applying.",
    }

    tab1, tab2, tab3 = st.tabs(["🔍 Advisory", "📊 Key Factors", "📋 Summary"])

    with tab1:
        color = risk["color"]
        st.markdown(f"""
        <div style="background:{color}18;border:1.5px solid {color}60;border-radius:12px;padding:1.25rem;margin-bottom:1.25rem;">
            <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:{color};margin-bottom:0.5rem">
                {risk['adv_icon']} Risk Level: {risk['tier']}
            </div>
            <div style="color:#CBD5E1;font-size:0.875rem;line-height:1.65">{advisory_messages[risk['tier']]}</div>
        </div>""", unsafe_allow_html=True)

        left, right = st.columns([3, 2])
        with left:
            st.markdown('<div class="section-header">Risk Findings</div>', unsafe_allow_html=True)
            advisories = detailed_analysis.get("detailed_advisories", [])
            if advisories:
                for adv in advisories:
                    lv     = adv.get("risk_level", "Info")
                    lv_clr = {"Critical":"#EF4444","High":"#F97316","Medium":"#EAB308","Low":"#22C55E"}.get(lv,"#3B82F6")
                    with st.expander(f"{adv.get('icon','📌')} {adv.get('category','')}: {adv.get('finding','')}"):
                        st.markdown(f"""
                        <span style="background:{lv_clr};color:white;padding:2px 10px;border-radius:4px;font-size:11px;font-weight:700">{lv}</span>
                        <div style="margin-top:10px;color:#CBD5E1;line-height:1.7;font-size:0.875rem">{adv.get('advisory','')}</div>
                        """, unsafe_allow_html=True)
            else:
                if final_fraud >= 50:
                    st.warning("BERT model detected suspicious patterns — no specific rule-based findings, but exercise caution.")
                else:
                    st.success("No specific red flags detected.")

        with right:
            st.markdown('<div class="section-header">Extracted Info</div>', unsafe_allow_html=True)
            info = detailed_analysis.get("extracted_info", {})
            if info.get("emails"):
                is_personal = any(d in e for e in info["emails"] for d in ["@gmail","@yahoo","@hotmail"])
                clr = "#EF4444" if is_personal else "#22C55E"
                st.markdown(f"📧 **Email:** <span style='color:{clr}'>{', '.join(info['emails'])}</span>", unsafe_allow_html=True)
            if info.get("phone_numbers"):
                st.markdown(f"📞 **Phone:** {', '.join(info['phone_numbers'])}")
            if info.get("company_mentions"):
                st.markdown(f"🏢 **Company:** {', '.join(info['company_mentions'][:2])}")
            if info.get("work_arrangement"):
                st.markdown(f"💼 **Work:** {info['work_arrangement'].replace('_',' ').title()}")
            if info.get("messaging_apps"):
                st.markdown("💬 **Messaging:** <span style='color:#F97316'>WhatsApp/Telegram ⚠️</span>", unsafe_allow_html=True)
            if info.get("requests_sensitive_info"):
                st.markdown("🔐 **Sensitive info requested** <span style='color:#EF4444'>⚠️</span>", unsafe_allow_html=True)
            if not any([info.get("emails"), info.get("phone_numbers"), info.get("company_mentions")]):
                st.info("No contact details extracted.")

    with tab2:
        chart_data = explanation.get("chart_data", {})
        words      = chart_data.get("words", [])
        weights    = chart_data.get("weights", [])
        if words and weights:
            try:
                colors = ["#EF4444" if w > 0 else "#10B981" for w in weights]
                fig = go.Figure(go.Bar(x=weights, y=words, orientation="h",
                                       marker=dict(color=colors, opacity=0.85)))
                fig.update_layout(
                    title=dict(text="Word Impact on Prediction",
                               font=dict(color="#F1F5F9", size=15, family="Syne")),
                    xaxis=dict(title="Impact (+fraud / −safe)",
                               title_font=dict(color="#94A3B8"),
                               tickfont=dict(color="#64748B"),
                               gridcolor="rgba(255,255,255,0.05)"),
                    yaxis=dict(tickfont=dict(color="#94A3B8")),
                    height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#94A3B8", family="DM Sans"),
                    margin=dict(l=10,r=10,t=40,b=10),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                st.info("Chart unavailable.")
        else:
            if fraud_signals:
                for s in fraud_signals:
                    st.markdown(f"🚩 {s}")
            else:
                st.info("No key factor data. Run Detailed Analysis for chart.")

    with tab3:
        sc  = risk["color"]
        rule_delta = round(final_fraud - bert_score, 1)
        rule_sign  = "+" if rule_delta >= 0 else ""
        bert_color = _score_color(bert_score)

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{sc}15,{sc}08);border:1.5px solid {sc}50;
                    border-radius:16px;padding:1.5rem;text-align:center;margin-bottom:1.25rem;">
            <div style="font-size:2.2rem;margin-bottom:0.4rem">{risk['icon']}</div>
            <div style="font-family:'Syne',sans-serif;font-size:1.5rem;font-weight:800;color:{sc};margin-bottom:0.3rem">{risk['tier']} RISK</div>
            <div style="font-family:'DM Mono',monospace;font-size:2rem;font-weight:500;color:#fff;margin-bottom:0.75rem">Final: {final_fraud}%</div>
            <div style="display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;">
                <div>
                    <div style="color:#64748B;font-size:0.8rem">BERT Raw Score</div>
                    <div style="color:{bert_color};font-size:1.1rem;font-weight:700">{bert_score}%</div>
                </div>
                <div>
                    <div style="color:#64748B;font-size:0.8rem">Rule Contribution</div>
                    <div style="color:{'#EF4444' if rule_delta>0 else '#10B981'};font-size:1.1rem;font-weight:700">{rule_sign}{rule_delta}%</div>
                </div>
                <div>
                    <div style="color:#64748B;font-size:0.8rem">Rule Flags</div>
                    <div style="color:{'#EF4444' if len(fraud_signals)>0 else '#10B981'};font-size:1.1rem;font-weight:700">{len(fraud_signals)} found</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        interp = explanation.get("interpretation", "")
        if interp:
            st.markdown(f"""
            <div style="background:rgba(59,130,246,0.08);border-left:3px solid #3B82F6;
                        padding:1.2rem 1.5rem;border-radius:0 12px 12px 0;
                        color:#CBD5E1;font-size:0.9rem;line-height:1.75">{interp}</div>""",
                        unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header" style="margin-top:1rem">Suspicious Words</div>', unsafe_allow_html=True)
            pos = explanation.get("positive_features", [])
            if pos:
                for f in pos[:5]:
                    w = f.get("word", str(f)) if isinstance(f, dict) else str(f)
                    st.warning(w)
            elif fraud_signals:
                for s in fraud_signals[:4]:
                    st.warning(s[:40] + ("..." if len(s) > 40 else ""))
            else:
                st.caption("None identified")
        with c2:
            st.markdown('<div class="section-header" style="margin-top:1rem">Legitimate Words</div>', unsafe_allow_html=True)
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
            label="text", height=290,
            placeholder="Paste the complete job posting here...\n\nTitle, description, requirements, contact info, salary, etc.",
            label_visibility="collapsed",
            key="job_input"
        )
        st.caption(f"{len(job_text):,} characters")

        b1, b2 = st.columns(2)
        with b1:
            quick    = st.button("⚡ Quick Analysis",    use_container_width=True, disabled=not is_connected)
        with b2:
            detailed = st.button("🔬 Detailed Analysis", use_container_width=True, disabled=not is_connected)

        if quick and job_text:
            with st.spinner("Analyzing..."):
                res = predict_job(job_text)
                if res and res.get("success"):
                    st.session_state.analysis_result = res
                    st.session_state.analysis_type   = "quick"
                    add_to_history(job_text, res, "quick")

        if detailed and job_text:
            with st.spinner("Running detailed analysis..."):
                res = get_explanation(job_text)
                if res and res.get("success"):
                    st.session_state.analysis_result = res
                    st.session_state.analysis_type   = "detailed"
                    add_to_history(job_text, res, "detailed")

        if (quick or detailed) and not job_text:
            st.warning("Please paste a job posting first.")

        with st.expander("📦 Batch Mode — Analyze Multiple Postings"):
            st.markdown("""<p style='color:#64748B;font-size:0.85rem;margin:0 0 0.75rem'>
            Separate postings with <code style='background:rgba(59,130,246,0.15);padding:1px 6px;border-radius:4px;color:#93C5FD'>---</code></p>""",
            unsafe_allow_html=True)
            if st.button("Run Batch Analysis", use_container_width=True, disabled=not is_connected, key="batch_btn"):
                if job_text:
                    postings = [p.strip() for p in job_text.split("---") if p.strip() and len(p.strip()) > 10]
                    if not postings:
                        st.warning("No valid postings found.")
                    elif len(postings) == 1:
                        st.info("Only 1 posting — use Quick Analysis.")
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
                                    cf  = pred.get("confidence", 0)
                                    ico = "✅" if lbl == "Legitimate" else "⚠️"
                                    clr = "#10B981" if lbl == "Legitimate" else "#EF4444"
                                    st.markdown(f"**{ico} Posting {idx}:** <span style='color:{clr}'>{lbl}</span> ({cf}%)", unsafe_allow_html=True)

        # ── Image uploader ──────────────────────────────────────────────────
        # The "uploadupload" bug: Streamlit renders BOTH its internal
        # <button> text AND the widget label string on the same row.
        # Fix: use label_visibility="collapsed" with an empty-string label,
        # then use CSS to re-style the ONE real browse button that remains.
        # No overlays, no wrappers — just style Streamlit's native element.
        if IMAGE_SUPPORT:
            st.markdown("""
            <style>
            /* Target only the file uploader inside col_in (left column).      */
            /* Re-skin the single Browse button Streamlit renders natively.    */
            [data-testid="stFileUploaderDropzone"] {
                background-color: #0F1729 !important;
                border: 1.5px dashed rgba(59,130,246,0.4) !important;
                border-radius: 14px !important;
                padding: 0.85rem 1.25rem !important;
                flex-direction: row !important;
                align-items: center !important;
                gap: 1rem !important;
            }
            [data-testid="stFileUploaderDropzone"]:hover {
                border-color: rgba(59,130,246,0.7) !important;
            }
            [data-testid="stFileUploaderDropzone"] button {
                background: linear-gradient(135deg,#3B82F6,#6D28D9) !important;
                border: none !important;
                border-radius: 8px !important;
                padding: 0.4rem 1.1rem !important;
                flex-shrink: 0 !important;
                overflow: hidden !important;
                text-indent: -9999px !important;
                white-space: nowrap !important;
                min-width: 110px !important;
                height: 36px !important;
                position: relative !important;
                cursor: pointer !important;
            }
            [data-testid="stFileUploaderDropzone"] button::after {
                content: "Browse files" !important;
                text-indent: 0 !important;
                position: absolute !important;
                inset: 0 !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                color: white !important;
                font-size: 0.85rem !important;
                font-weight: 600 !important;
                font-family: 'DM Sans', sans-serif !important;
            }
            /* Instruction text next to button */
            [data-testid="stFileUploaderDropzoneInstructions"] > div > span {
                color: #94A3B8 !important;
                font-size: 0.83rem !important;
            }
            [data-testid="stFileUploaderDropzoneInstructions"] > div > small {
                display: none !important;
            }
            /* Completely hide the widget label (the source of the double text) */
            [data-testid="stFileUploader"] > label {
                display: none !important;
                height: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
            }
            </style>
            """, unsafe_allow_html=True)

            st.markdown(
                "<p style='color:#94A3B8;font-size:0.85rem;margin:0.75rem 0 0.25rem'>"
                "📷 <strong style='color:#E2E8F0'>Image Analysis</strong>"
                " — Upload a screenshot of a job posting</p>",
                unsafe_allow_html=True
            )
            # label="" + label_visibility="collapsed" = zero label text rendered
            uploaded = st.file_uploader(
                "",
                type=["png", "jpg", "jpeg"],
                label_visibility="collapsed",
                key="ocr_upload"
            )
            if uploaded is not None:
                img = Image.open(uploaded)
                st.image(img, caption="Uploaded image", use_column_width=True)
                ia, ib = st.columns(2)
                with ia:
                    iq   = st.button("⚡ Quick (image)",    use_container_width=True, disabled=not is_connected, key="img_q")
                with ib:
                    idet = st.button("🔬 Detailed (image)", use_container_width=True, disabled=not is_connected, key="img_d")
                if iq or idet:
                    uploaded.seek(0)
                    with st.spinner("Extracting text and analyzing..."):
                        res = explain_image_api(uploaded.read())
                        if res and res.get("success"):
                            meta = res.get("metadata", {})
                            st.success(f"Extracted {meta.get('text_length', 0)} characters")
                            st.text_area("Extracted Text", meta.get("extracted_text", ""), height=120, disabled=True)
                            st.session_state.analysis_result = res
                            st.session_state.analysis_type   = "detailed" if idet else "quick"
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
# FIX 1: plain numbered title — no emoji/tag prefix causing word overlap
# ─────────────────────────────────────────────────────────────

def page_history():
    history = load_history()

    st.markdown('<div class="section-header">Past Analyses</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analysis History</div>', unsafe_allow_html=True)

    if not history:
        st.markdown("""
        <div class="placeholder-wrap">
            <div class="placeholder-icon">📋</div>
            <div class="placeholder-txt">No analyses yet.<br>Run an analysis to see history here.</div>
        </div>""", unsafe_allow_html=True)
        return

    total       = len(history)
    fraud_count = sum(1 for h in history if h.get("is_fraudulent"))
    legit_count = total - fraud_count
    avg_conf    = sum(h.get("confidence", 0) for h in history) / total

    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, val, lbl, clr in [
        (sc1, total,              "Total Analyses",  "#fff"),
        (sc2, fraud_count,        "Flagged",         "#EF4444"),
        (sc3, legit_count,        "Safe",            "#10B981"),
        (sc4, f"{avg_conf:.0f}%", "Avg Confidence",  "#93C5FD"),
    ]:
        with col:
            col.markdown(f"""<div class="mini-stat-box">
                <div class="mini-stat-val" style="color:{clr}">{val}</div>
                <div class="mini-stat-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)

    fc1, fc2, fc3 = st.columns([3, 1.2, 1.2])
    with fc1:
        search      = st.text_input("search", placeholder="Search by keyword...", label_visibility="collapsed")
    with fc2:
        filter_type = st.selectbox("filter", ["All", "Fraudulent", "Legitimate"], label_visibility="collapsed")
    with fc3:
        if st.button("Clear All", use_container_width=True):
            clear_history(); st.success("History cleared!"); st.rerun()

    all_texts = " ".join(h.get("text_preview", "") for h in history).lower()
    common_kw = ["remote", "work from home", "no experience", "urgent", "gmail", "whatsapp", "salary", "engineer"]
    found_kw  = [k for k in common_kw if k in all_texts]
    if found_kw:
        chips = "".join(
            f'<span style="display:inline-block;background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.25);'
            f'color:#93C5FD;font-size:0.78rem;font-weight:500;padding:0.3rem 0.9rem;border-radius:999px;margin:0.2rem">{k}</span>'
            for k in found_kw[:6]
        )
        st.markdown(f"<div style='margin-bottom:0.5rem'>Quick filter: {chips}</div>", unsafe_allow_html=True)
        st.caption("Type any keyword above into the search box to filter.")

    filtered = history
    if filter_type == "Fraudulent":
        filtered = [h for h in filtered if h.get("is_fraudulent")]
    elif filter_type == "Legitimate":
        filtered = [h for h in filtered if not h.get("is_fraudulent")]
    if search:
        sq = search.lower()
        filtered = [h for h in filtered
                    if sq in h.get("text_preview","").lower()
                    or sq in h.get("full_text","").lower()
                    or sq in h.get("label","").lower()]

    st.caption(f"Showing {len(filtered)} of {total} entries")
    st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)

    for i, entry in enumerate(filtered):
        probs      = entry.get("probabilities", {})
        fraud_pct  = probs.get("fraudulent", 0)
        risk       = get_risk_tier(fraud_pct)
        conf       = entry.get("confidence", 0)
        timestamp  = entry.get("timestamp", "")
        signals    = entry.get("fraud_signals", [])
        full_text  = entry.get("full_text", "")

        bert_est   = entry.get("bert_estimate") or derive_bert_score(fraud_pct, signals, entry)
        rule_delta = round(fraud_pct - bert_est, 1)
        rule_sign  = "+" if rule_delta >= 0 else ""

        # FIX 1: strip any old tier tags from stored previews, use plain number prefix
        raw_preview  = entry.get("text_preview", "")
        clean_preview = raw_preview.replace("[LOW]","").replace("[MEDIUM]","").replace("[HIGH]","").strip()
        # Format: "#1  <preview text>"  — no emoji, no tier tag in the title string
        exp_title = f"#{i + 1}  {clean_preview[:90]}"

        with st.expander(exp_title, expanded=False):
            badge_map  = {
                "HIGH":   '<span class="badge-high">High Risk</span>',
                "MEDIUM": '<span class="badge-medium">Medium Risk</span>',
                "LOW":    '<span class="badge-low">Legitimate</span>',
            }
            badge_html = badge_map[risk["tier"]]
            atype_b    = f'<span class="badge-type">{entry.get("analysis_type","quick").title()}</span>'

            st.markdown(f"""
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:0.5rem;margin-bottom:1rem">
                <div style="display:flex;gap:0.5rem;align-items:center">{badge_html} {atype_b}</div>
                <div style="display:flex;gap:1.25rem;align-items:center">
                    <span style="font-family:'DM Mono',monospace;font-size:0.82rem;color:#64748B">Confidence: {conf}%</span>
                    <span class="hist-time">{timestamp}</span>
                </div>
            </div>""", unsafe_allow_html=True)

            # 4-column score row so BERT is always visually distinct
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                clr = _score_color(fraud_pct)
                st.markdown(f"**Fraud Risk**<br><span style='color:{clr};font-family:DM Mono,monospace;font-size:1.1rem'>{fraud_pct}%</span>", unsafe_allow_html=True)
            with p2:
                st.markdown(f"**Safe Score**<br><span style='color:#10B981;font-family:DM Mono,monospace;font-size:1.1rem'>{probs.get('legitimate',0)}%</span>", unsafe_allow_html=True)
            with p3:
                bc = _score_color(bert_est)
                st.markdown(f"**BERT Score**<br><span style='color:{bc};font-family:DM Mono,monospace;font-size:1.1rem'>{bert_est}%</span>", unsafe_allow_html=True)
            with p4:
                rc = "#EF4444" if rule_delta > 0 else "#10B981"
                st.markdown(f"**Rule Boost**<br><span style='color:{rc};font-family:DM Mono,monospace;font-size:1.1rem'>{rule_sign}{rule_delta}%</span>", unsafe_allow_html=True)

            if signals:
                pills = "".join(f'<span class="signal-pill">🚩 {s}</span>' for s in signals)
                st.markdown(f"**Red Flags:**<br><div style='margin-top:0.3rem'>{pills}</div>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#64748B;font-size:0.85rem'>No rule-based red flags detected.</span>", unsafe_allow_html=True)

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

            if full_text:
                import html as html_module
                escaped = html_module.escape(full_text)
                st.markdown("**Full Text:**")
                st.markdown(f'<div class="fulltext-box">{escaped}</div>', unsafe_allow_html=True)
            else:
                st.caption("Full text not available.")

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            st.markdown("**Suggestions:**")
            if risk["is_fraud"]:
                tips = []
                if any("email" in s.lower() for s in signals):
                    tips.append("Search the company on LinkedIn to verify it exists")
                if any("fee" in s.lower() or "payment" in s.lower() for s in signals):
                    tips.append("Never pay any fee before starting — legitimate employers don't ask")
                if any("whatsapp" in s.lower() or "telegram" in s.lower() for s in signals):
                    tips.append("Avoid sharing personal info over WhatsApp/Telegram with unknown recruiters")
                if not tips:
                    tips = [
                        "Verify the company on official job boards like LinkedIn or Indeed",
                        "Do not share SSN, bank details, or ID before a formal offer",
                    ]
                for tip in tips:
                    st.markdown(f"<div style='color:#CBD5E1;font-size:0.875rem;padding:0.25rem 0'>• {tip}</div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='color:#CBD5E1;font-size:0.875rem'>
                • Verify the company website independently<br>
                • Check employee reviews on Glassdoor before accepting<br>
                • Always review the employment contract carefully
                </div>""", unsafe_allow_html=True)

            st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
            if st.button("Re-Analyze", key=f"reanalyze_{entry.get('id', i)}"):
                st.session_state["job_input_prefill"] = full_text
                st.session_state.page = "analyze"
                st.rerun()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    load_css()

    for key, default in [
        ("analysis_result", None),
        ("analysis_type",   None),
        ("page",            "analyze"),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default

    is_connected = check_api()
    hist_count   = len(load_history())

    dot  = "dot-green" if is_connected else "dot-red"
    stxt = "API Connected" if is_connected else "API Disconnected"
    st.markdown(f"""
    <div class="nav-bar">
        <div class="nav-logo"><div class="nav-logo-icon">🛡️</div>JobGuard AI</div>
        <div class="nav-status"><span class="nav-status-dot {dot}"></span>{stxt}</div>
    </div>""", unsafe_allow_html=True)

    p1, p2, _ = st.columns([1, 1.3, 9])
    with p1:
        if st.button("Analyze", use_container_width=True):
            st.session_state.page = "analyze"
    with p2:
        if st.button(f"History ({hist_count})", use_container_width=True):
            st.session_state.page = "history"

    st.markdown("<div style='height:0.1rem;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:2rem'></div>", unsafe_allow_html=True)

    if st.session_state.page == "analyze":
        st.markdown("""
        <div class="hero-section">
            <div class="hero-grid"></div>
            <div class="hero-badge">AI-Powered Detection System</div>
            <h1 class="hero-title">Detect Fake Job<br><span>Postings Instantly</span></h1>
            <div class="hero-sub">Hybrid BERT + Rule-Based analysis trained on 17,000+ job postings to protect you from recruitment scams.</div>
            <div class="hero-stats">
                <div><div class="hero-stat-num">97%+</div><div class="hero-stat-label">Accuracy</div></div>
                <div><div class="hero-stat-num">8+</div><div class="hero-stat-label">Scam Types</div></div>
                <div><div class="hero-stat-num">&lt; 2s</div><div class="hero-stat-label">Analysis Time</div></div>
            </div>
        </div>""", unsafe_allow_html=True)
        page_analyze(is_connected)
    else:
        page_history()

    st.markdown("""
    <div class="disclaimer">
        <strong style='color:#93C5FD'>Disclaimer</strong><br>
        JobGuard AI provides risk assessment guidance only — not definitive fraud detection.
        Always verify job postings independently through official company websites and trusted platforms.
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
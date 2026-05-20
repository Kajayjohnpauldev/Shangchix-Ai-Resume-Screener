"""Shangchix AI Resume — Streamlit frontend.

Run with: streamlit run frontend/app.py
The backend must be reachable at BACKEND_URL (default http://127.0.0.1:8000).
"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Shangchix AI Resume",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom theme / CSS -----------------------------------------------------
st.markdown(
    """
    <style>
    /* Hide Streamlit's default header padding so our banner sits flush. */
    .block-container { padding-top: 2rem; padding-bottom: 3rem; }

    /* Gradient hero banner */
    .shx-hero {
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 45%, #ec4899 100%);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        box-shadow: 0 10px 30px rgba(79, 70, 229, 0.25);
        margin-bottom: 28px;
    }
    .shx-hero h1 {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .shx-hero p {
        margin: 6px 0 0 0;
        opacity: 0.92;
        font-size: 1rem;
    }
    .shx-hero .shx-tag {
        display: inline-block;
        background: rgba(255,255,255,0.18);
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
        margin-top: 10px;
    }

    /* Candidate cards */
    .shx-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .shx-card-header {
        display: flex; align-items: center; justify-content: space-between;
        gap: 12px; margin-bottom: 8px;
    }
    .shx-rank {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        width: 38px; height: 38px;
        border-radius: 50%;
        display: inline-flex; align-items: center; justify-content: center;
        font-weight: 700;
    }
    .shx-name { font-size: 1.15rem; font-weight: 700; color: #111827; }
    .shx-score-pill {
        background: #ecfdf5; color: #047857;
        padding: 4px 12px; border-radius: 999px;
        font-weight: 700; font-size: 0.9rem;
    }
    .shx-score-pill.med { background: #fef3c7; color: #92400e; }
    .shx-score-pill.low { background: #fee2e2; color: #991b1b; }

    /* Skill / gap chips */
    .shx-chip {
        display: inline-block;
        padding: 4px 10px;
        margin: 3px 4px 3px 0;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .shx-chip.match { background: #dcfce7; color: #166534; }
    .shx-chip.gap   { background: #fee2e2; color: #991b1b; }

    .shx-section-label {
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-size: 0.72rem;
        font-weight: 700;
        color: #6b7280;
        margin: 14px 0 4px 0;
    }
    .shx-explain {
        background: #f5f3ff;
        border-left: 4px solid #7c3aed;
        padding: 12px 14px;
        border-radius: 8px;
        font-size: 0.95rem;
        color: #1f2937;
        line-height: 1.55;
    }

    /* Score-button primary tweak */
    .stButton > button[kind="primary"] {
        background: linear-gradient(120deg, #4f46e5, #7c3aed);
        border: 0;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(120deg, #4338ca, #6d28d9);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Hero banner ------------------------------------------------------------
st.markdown(
    """
    <div class="shx-hero">
      <h1>🧠 Shangchix AI Resume</h1>
      <p>Rank candidates against any job description with semantic search and explainable AI reasoning.</p>
      <div>
        <span class="shx-tag">RAG</span>
        <span class="shx-tag">FAISS</span>
        <span class="shx-tag">Sentence-Transformers</span>
        <span class="shx-tag">Gemini · LiteLLM</span>
        <span class="shx-tag">FastAPI · Streamlit</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        "1. PDFs → text via **pypdf**\n"
        "2. JD + resumes → embeddings (**MiniLM-L6**)\n"
        "3. **FAISS** cosine similarity → 0–100 score\n"
        "4. Top-K candidates → **LLM** recruiter blurb"
    )
    st.divider()
    st.markdown("### Tips")
    st.markdown(
        "- Paste the **full JD** for best semantic match.\n"
        "- 5–20 resumes is the sweet spot.\n"
        "- Use the bundled `tests/sample_resumes/` PDFs for a quick demo."
    )
    st.divider()
    st.caption(f"Backend: `{BACKEND_URL}`")
    try:
        h = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if h.status_code == 200:
            st.success("Backend online ✅", icon="🟢")
        else:
            st.warning(f"Backend status {h.status_code}", icon="⚠️")
    except requests.exceptions.RequestException:
        st.error("Backend unreachable", icon="🔴")


# --- Inputs -----------------------------------------------------------------
col_jd, col_files = st.columns([3, 2], gap="large")

with col_jd:
    st.markdown("#### 📝 Job Description")
    job_description = st.text_area(
        "Job Description",
        label_visibility="collapsed",
        height=260,
        placeholder=(
            "Paste the JD here. Example:\n\n"
            "We are hiring a backend engineer with strong Python skills, "
            "experience building REST APIs with FastAPI or Flask, "
            "familiarity with Docker and AWS, and exposure to ML/LLM "
            "pipelines (sentence-transformers, FAISS, RAG)."
        ),
    )

with col_files:
    st.markdown("#### 📄 Resumes (PDF)")
    uploaded = st.file_uploader(
        "Resumes",
        label_visibility="collapsed",
        type=["pdf"],
        accept_multiple_files=True,
        help="Drop one or many PDFs. Bigger batches still finish in under a minute.",
    )
    if uploaded:
        st.caption(f"📎 {len(uploaded)} resume(s) loaded")

st.write("")
go = st.button(
    "🚀 Score Resumes",
    type="primary",
    use_container_width=True,
    disabled=not (job_description and uploaded),
)


# --- Helpers ----------------------------------------------------------------
def _score_class(score: float) -> str:
    if score >= 65:
        return ""
    if score >= 45:
        return "med"
    return "low"


def _chip(text: str, kind: str) -> str:
    return f'<span class="shx-chip {kind}">{text}</span>'


# --- Run ranking ------------------------------------------------------------
if go:
    with st.spinner(f"Scoring {len(uploaded)} resume(s)…"):
        files = [
            ("resumes", (f.name, f.getvalue(), "application/pdf"))
            for f in uploaded
        ]
        data = {"job_description": job_description}
        t0 = time.time()
        try:
            resp = requests.post(
                f"{BACKEND_URL}/score",
                data=data,
                files=files,
                timeout=180,
            )
        except requests.exceptions.RequestException as exc:
            st.error(f"Could not reach backend: {exc}")
            st.stop()
        elapsed = time.time() - t0

    if resp.status_code != 200:
        st.error(f"Backend returned {resp.status_code}: {resp.text}")
        st.stop()

    payload = resp.json()
    results = payload["results"]

    # Summary metrics row
    top_score = results[0]["score"] if results else 0
    avg_score = (
        sum(r["score"] for r in results) / len(results) if results else 0
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates", payload["candidate_count"])
    m2.metric("Top score", f"{top_score:.1f}")
    m3.metric("Average", f"{avg_score:.1f}")
    m4.metric("Elapsed", f"{elapsed:.1f}s")

    st.markdown("### 🏆 Ranked candidates")

    for rank_idx, r in enumerate(results, start=1):
        score = float(r["score"])
        score_cls = _score_class(score)
        matched_chips = "".join(
            _chip(s.title(), "match") for s in r["top_skills_matched"]
        ) or '<span style="color:#6b7280;">— none surfaced —</span>'
        gap_chips = "".join(
            _chip(s.title(), "gap") for s in r["top_gaps"]
        ) or '<span style="color:#6b7280;">— none surfaced —</span>'

        st.markdown(
            f"""
            <div class="shx-card">
              <div class="shx-card-header">
                <div style="display:flex; align-items:center; gap:14px;">
                  <span class="shx-rank">{rank_idx}</span>
                  <span class="shx-name">{r['candidate_name']}</span>
                </div>
                <span class="shx-score-pill {score_cls}">{score:.1f} / 100</span>
              </div>
              <div style="color:#6b7280; font-size:0.85rem; margin-bottom:6px;">
                Cosine similarity: {r['raw_similarity']:.3f}
              </div>
              <div class="shx-section-label">Top matching skills</div>
              <div>{matched_chips}</div>
              <div class="shx-section-label">Top gaps vs. JD</div>
              <div>{gap_chips}</div>
              <div class="shx-section-label">AI explanation</div>
              <div class="shx-explain">{r['explanation']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(int(score), 100))

else:
    st.info(
        "👆 Paste a job description and upload one or more PDF resumes, "
        "then hit **Score Resumes**.",
        icon="💡",
    )

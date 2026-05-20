"""Shangchix AI Resume — Streamlit frontend (minimal black & white).

Run with: streamlit run frontend/app.py
The backend must be reachable at BACKEND_URL (default http://127.0.0.1:8000).

Results are rendered with native Streamlit widgets (st.container, st.metric,
st.markdown) rather than hand-rolled HTML — that keeps the text crisp and
always visible, and lets the theme in .streamlit/config.toml own the look.
"""
from __future__ import annotations

import os
import time

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Shangchix AI Resume",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Tiny bit of CSS: black progress bars + a touch more contrast. Everything
# else comes from the theme config (black/white/grey) so it stays clean.
st.markdown(
    """
    <style>
    .block-container { max-width: 1080px; padding-top: 2rem; }
    .stProgress > div > div > div > div { background-color: #111111; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Header -----------------------------------------------------------------
st.title("Shangchix AI Resume")
st.caption(
    "Rank candidates against a job description with semantic search, "
    "an ATS keyword score, and concrete fixes to raise each resume."
)
st.divider()


# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        "1. PDF / Word → text\n"
        "2. JD + resumes → embeddings (**MiniLM-L6**)\n"
        "3. **FAISS** cosine similarity → match score\n"
        "4. JD keyword coverage → **ATS score**\n"
        "5. Top candidates → **LLM** assessment + fixes"
    )
    st.divider()
    st.markdown("### Tips")
    st.markdown(
        "- Paste the **full JD** for the best match.\n"
        "- Upload **PDF or .docx** resumes.\n"
        "- Use the bundled `tests/sample_resumes/` files for a quick demo."
    )
    st.divider()
    st.caption(f"Backend: `{BACKEND_URL}`")
    try:
        h = requests.get(f"{BACKEND_URL}/health", timeout=2)
        if h.status_code == 200:
            st.success("Backend online", icon="✅")
        else:
            st.warning(f"Backend status {h.status_code}", icon="⚠️")
    except requests.exceptions.RequestException:
        st.error("Backend unreachable", icon="🔴")


# --- Inputs -----------------------------------------------------------------
col_jd, col_files = st.columns([3, 2], gap="large")

with col_jd:
    st.markdown("#### Job Description")
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
    st.markdown("#### Resumes (PDF or Word)")
    uploaded = st.file_uploader(
        "Resumes",
        label_visibility="collapsed",
        type=["pdf", "docx"],
        accept_multiple_files=True,
        help="Drop one or many PDF or .docx files.",
    )
    if uploaded:
        st.caption(f"{len(uploaded)} resume(s) loaded")

st.write("")
go = st.button("Score Resumes", type="primary", use_container_width=True)


# --- Run ranking ------------------------------------------------------------
if go:
    missing = []
    if not job_description:
        missing.append("a **job description**")
    if not uploaded:
        missing.append("at least **one PDF or Word resume**")
    if missing:
        st.warning("Add " + " and ".join(missing) + " first, then hit Score.", icon="⚠️")
        st.stop()

    with st.spinner(f"Scoring {len(uploaded)} resume(s)…"):
        files = [
            ("resumes", (f.name, f.getvalue(), f.type or "application/octet-stream"))
            for f in uploaded
        ]
        t0 = time.time()
        try:
            resp = requests.post(
                f"{BACKEND_URL}/score",
                data={"job_description": job_description},
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
    avg_ats = (
        sum(r.get("ats_score", 0) for r in results) / len(results)
        if results else 0
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Candidates", payload["candidate_count"])
    m2.metric("Top match", f"{top_score:.0f}/100")
    m3.metric("Avg ATS", f"{avg_ats:.0f}%")
    m4.metric("Elapsed", f"{elapsed:.1f}s")

    st.subheader("Ranked candidates")

    for rank_idx, r in enumerate(results, start=1):
        score = float(r["score"])
        ats = float(r.get("ats_score", 0.0))
        projected = float(r.get("projected_score", score))
        uplift = float(r.get("score_uplift", 0.0))
        kw_m = int(r.get("keywords_matched", 0))
        kw_t = int(r.get("keywords_total", 0))

        with st.container(border=True):
            head_l, head_r = st.columns([4, 1])
            with head_l:
                st.markdown(f"### {rank_idx}.  {r['candidate_name']}")
            with head_r:
                st.markdown(
                    f"<div style='text-align:right;font-size:1.6rem;"
                    f"font-weight:800;'>{score:.0f}"
                    f"<span style='font-size:0.8rem;color:#888;'>/100</span></div>",
                    unsafe_allow_html=True,
                )

            # Score dashboard
            c1, c2, c3 = st.columns(3)
            c1.metric("Match score", f"{score:.0f}/100")
            c2.metric(
                "ATS keyword score",
                f"{ats:.0f}%",
                help=f"{kw_m} of {kw_t} JD keywords found in this resume",
            )
            c3.metric(
                "Projected match",
                f"{projected:.0f}/100",
                delta=(f"+{uplift:.0f}" if uplift >= 0.5 else None),
                help="Score if the candidate adds the missing skills below",
            )
            st.progress(min(int(score), 100))

            # Skills — show the COMPLETE lists so the chips line up with the
            # ATS keyword count above (e.g. ATS 7/8 → 7 matching chips).
            matched = r.get("all_skills_matched") or r["top_skills_matched"]
            gaps = r.get("all_gaps") or r["top_gaps"]
            sk_l, sk_r = st.columns(2)
            with sk_l:
                st.markdown(f"**✅ Matching skills ({len(matched)})**")
                st.markdown(
                    " ".join(f"`{s}`" for s in matched) if matched
                    else "_none surfaced_"
                )
            with sk_r:
                st.markdown(f"**⬜ Missing skills to add ({len(gaps)})**")
                st.markdown(
                    " ".join(f"`{s}`" for s in gaps) if gaps
                    else "_none surfaced_"
                )

            # AI assessment
            st.markdown("**🧠 AI assessment**")
            st.info(r["explanation"], icon="📝")

            # Improvements
            improvements = r.get("improvements", []) or []
            if improvements:
                st.markdown("**🔧 How to improve this resume**")
                st.markdown("\n".join(f"- {t}" for t in improvements))

            st.caption(f"Cosine similarity: {r['raw_similarity']:.3f}")

else:
    st.info(
        "Paste a job description and upload one or more PDF or Word resumes, "
        "then click **Score Resumes**.",
        icon="💡",
    )

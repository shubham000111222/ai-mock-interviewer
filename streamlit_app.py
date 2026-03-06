"""
AI Mock Interviewer — Streamlit Application
===========================================
Flow:
  1. Upload / paste a Job Description
  2. Configure interview settings
  3. AI generates role-specific questions
  4. For each question: record / upload audio answer
  5. Whisper transcribes the audio
  6. Feedback Engine evaluates content quality + confidence
  7. Dashboard shows per-question scores + overall report

Run:
    streamlit run streamlit_app.py
"""

import os
import sys
import time
import json
import tempfile
import base64
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Adjust path so `src` is importable
sys.path.insert(0, str(Path(__file__).parent))

from src.question_generator import get_question_generator
from src.speech_processor import SpeechProcessor
from src.feedback_engine import FeedbackEngine

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Mock Interviewer",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .score-card {
        border-radius: 12px;
        padding: 16px 24px;
        text-align: center;
        margin: 8px 0;
    }
    .score-high   { background: linear-gradient(135deg,#1a9c3e,#27ae60); color:white; }
    .score-medium { background: linear-gradient(135deg,#d68910,#f39c12); color:white; }
    .score-low    { background: linear-gradient(135deg,#a93226,#e74c3c); color:white; }
    .question-card {
        border-left: 4px solid #667eea;
        padding: 12px 20px;
        margin: 12px 0;
        background: #f8f9ff;
        border-radius: 0 8px 8px 0;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 2px;
    }
    .badge-tech   { background:#dbeafe; color:#1d4ed8; }
    .badge-behav  { background:#dcfce7; color:#15803d; }
    .badge-sit    { background:#fef9c3; color:#92400e; }
    .badge-easy   { background:#f0fdf4; color:#166534; }
    .badge-medium { background:#fff7ed; color:#c2410c; }
    .badge-hard   { background:#fef2f2; color:#991b1b; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session State ──────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "stage": "setup",           # setup | interview | report
        "questions": [],
        "current_q": 0,
        "results": [],
        "job_description": "",
        "backend_llm": "mock",
        "backend_stt": "mock",
        "content_backend": "rules",
        "num_questions": 8,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/microphone.png", width=72)
    st.title("⚙️ Settings")
    st.divider()

    st.subheader("AI Backend")
    llm_choice = st.selectbox(
        "Question Generator",
        ["mock (demo)", "openai (GPT-4o-mini)", "ollama (Llama3)"],
        help="'mock' works offline; others need API/Ollama running",
    )
    st.session_state.backend_llm = llm_choice.split(" ")[0]

    stt_choice = st.selectbox(
        "Speech-to-Text",
        ["mock (demo)", "local_whisper (base)", "local_whisper (small)", "openai_whisper"],
        help="'mock' returns sample text; local_whisper needs ffmpeg + openai-whisper",
    )
    st.session_state.backend_stt = stt_choice.split(" ")[0]
    whisper_model = "base" if "base" in stt_choice else "small"

    content_backend = st.selectbox(
        "Feedback Engine",
        ["rules (offline)", "llm (GPT)"],
    )
    st.session_state.content_backend = content_backend.split(" ")[0]

    st.divider()
    st.subheader("Interview Config")
    st.session_state.num_questions = st.slider("Number of Questions", 4, 15, 8)
    num_tech = st.slider("Technical", 1, 8, 4)
    num_beh  = st.slider("Behavioral", 1, 5, 2)
    num_sit  = st.slider("Situational", 1, 5, 2)

    st.divider()
    if st.button("🔄 Restart Interview", use_container_width=True, type="secondary"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ── Helper functions ───────────────────────────────────────────────────────────
def score_color(score: int) -> str:
    if score >= 75: return "score-high"
    if score >= 50: return "score-medium"
    return "score-low"

def badge(text: str, kind: str) -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'

def category_badge(cat: str) -> str:
    k = {"Technical": "tech", "Behavioral": "behav", "Situational": "sit"}.get(cat, "tech")
    return badge(cat, k)

def difficulty_badge(diff: str) -> str:
    k = diff.lower()
    return badge(diff, k)

@st.cache_resource
def get_processor(stt_backend: str, whisper_size: str):
    return SpeechProcessor(backend=stt_backend, whisper_model=whisper_size)

@st.cache_resource
def get_engine(content_backend: str):
    return FeedbackEngine(content_backend=content_backend)


# ── Stage: SETUP ───────────────────────────────────────────────────────────────
if st.session_state.stage == "setup":
    st.title("🎙️ AI Mock Interviewer")
    st.markdown(
        "Upload a job description → AI generates tailored questions → "
        "Record your answers → Get instant feedback on **content quality** and **confidence**."
    )
    st.divider()

    col1, col2 = st.columns([3, 2])

    with col1:
        st.subheader("📋 Job Description")
        tab_paste, tab_upload = st.tabs(["✏️ Paste Text", "📁 Upload File"])

        with tab_paste:
            jd = st.text_area(
                "Paste the job description here",
                height=320,
                placeholder="e.g. We are looking for a Senior Data Scientist with 3+ years of experience in ML, Python, NLP...",
                value=st.session_state.job_description,
            )
            st.session_state.job_description = jd

        with tab_upload:
            f = st.file_uploader("Upload .txt or .pdf", type=["txt", "pdf"])
            if f:
                if f.type == "application/pdf":
                    try:
                        import pdfplumber
                        with pdfplumber.open(f) as pdf:
                            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
                    except ImportError:
                        st.warning("Install pdfplumber: `pip install pdfplumber`")
                        text = ""
                else:
                    text = f.read().decode("utf-8", errors="ignore")
                st.session_state.job_description = text
                st.success(f"Loaded {len(text.split())} words from {f.name}")

    with col2:
        st.subheader("🚀 Quick Start")
        st.info(
            "**1.** Paste a job description on the left.\n\n"
            "**2.** Adjust settings in the sidebar.\n\n"
            "**3.** Click **Generate Questions** below.\n\n"
            "**4.** Record your answers and get AI feedback!"
        )

        # Sample JDs
        st.subheader("📌 Sample Job Descriptions")
        samples = {
            "Data Scientist": (
                "We are seeking a Data Scientist with 2+ years of experience. "
                "Skills: Python, scikit-learn, SQL, pandas, data visualization (Matplotlib/Seaborn/Plotly). "
                "Experience with NLP, deep learning (PyTorch/TensorFlow) is a plus. "
                "You will build predictive models, run A/B tests, and collaborate with engineering teams. "
                "Strong communication skills required to present insights to stakeholders."
            ),
            "ML Engineer": (
                "ML Engineer role requiring expertise in deploying ML models at scale. "
                "Tech: Python, Docker, Kubernetes, FastAPI, MLflow, CI/CD. "
                "Experience with feature stores, model monitoring, and distributed training. "
                "Familiarity with AWS/GCP and their ML services (SageMaker/Vertex AI) preferred."
            ),
            "Software Engineer": (
                "Backend Software Engineer — Java/Python, microservices, REST APIs, PostgreSQL, Redis. "
                "Experience with system design, high availability, and distributed systems. "
                "Comfortable with Agile, code reviews, and working in cross-functional teams."
            ),
        }
        for title, desc in samples.items():
            if st.button(f"Use: {title}", use_container_width=True):
                st.session_state.job_description = desc
                st.rerun()

    st.divider()
    if st.button("⚡ Generate Interview Questions", type="primary", use_container_width=True):
        if not st.session_state.job_description.strip():
            st.error("Please provide a job description first.")
        else:
            with st.spinner("🤖 Generating tailored interview questions..."):
                generator = get_question_generator(st.session_state.backend_llm)
                questions = generator.generate(
                    job_description=st.session_state.job_description,
                    num_questions=st.session_state.num_questions,
                    num_technical=num_tech,
                    num_behavioral=num_beh,
                    num_situational=num_sit,
                )
            st.session_state.questions = questions
            st.session_state.current_q = 0
            st.session_state.results = []
            st.session_state.stage = "interview"
            st.rerun()


# ── Stage: INTERVIEW ───────────────────────────────────────────────────────────
elif st.session_state.stage == "interview":
    questions = st.session_state.questions
    idx = st.session_state.current_q
    total = len(questions)

    # Progress bar
    st.progress((idx) / total, text=f"Question {idx + 1} of {total}")

    q = questions[idx]

    # Question card
    st.markdown(
        f"""
        <div class="question-card">
          <div style="margin-bottom:8px">
            {category_badge(q.get('category','Technical'))}
            {difficulty_badge(q.get('difficulty','Medium'))}
          </div>
          <h3 style="margin:0">Q{q.get('id',idx+1)}: {q['question']}</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if q.get("follow_up"):
        with st.expander("💬 Potential follow-up"):
            st.write(q["follow_up"])

    st.divider()

    col_rec, col_hint = st.columns([3, 2])

    with col_rec:
        st.subheader("🎙️ Your Answer")
        answer_tab1, answer_tab2 = st.tabs(["🎤 Upload Audio", "⌨️ Type Answer"])

        with answer_tab1:
            st.info(
                "Record your answer using any voice recorder app, then upload the audio file here.\n\n"
                "_Supported: WAV, MP3, M4A, WEBM_"
            )
            audio_file = st.file_uploader(
                "Upload your recorded answer",
                type=["wav", "mp3", "m4a", "webm", "ogg"],
                key=f"audio_{idx}",
            )
            if audio_file:
                st.audio(audio_file)

        with answer_tab2:
            typed_answer = st.text_area(
                "Type your answer here (will skip audio analysis)",
                height=180,
                key=f"typed_{idx}",
                placeholder="Start your answer...",
            )

    with col_hint:
        st.subheader("💡 Hints")
        if q.get("ideal_answer_hints"):
            for hint in q["ideal_answer_hints"]:
                st.markdown(f"- {hint}")
        else:
            st.write("No hints available.")

        st.subheader("⏱️ Speaking Tips")
        st.markdown(
            """
            - Aim for **60-120 seconds**
            - Use **STAR** method for behavioral questions
            - Be specific — mention **numbers/outcomes**
            - Avoid filler words (*um, uh, like*)
            - Speak at a **steady, confident pace**
            """
        )

    st.divider()

    col_skip, col_submit = st.columns([1, 3])

    with col_skip:
        if st.button("⏭️ Skip Question"):
            if idx + 1 < total:
                st.session_state.current_q += 1
            else:
                st.session_state.stage = "report"
            st.rerun()

    with col_submit:
        submit_label = "📊 Submit & Get Feedback" if idx + 1 < total else "📊 Submit & View Report"
        if st.button(submit_label, type="primary"):
            processor = get_processor(st.session_state.backend_stt, whisper_model)
            engine = get_engine(st.session_state.content_backend)

            with st.spinner("🔍 Analysing your answer..."):
                # Determine audio path or use typed text
                if audio_file:
                    suffix = Path(audio_file.name).suffix
                    audio_path = processor.save_audio_bytes(audio_file.read(), suffix=suffix)
                    speech_result = processor.process(audio_path)
                    transcript = speech_result["transcript"]
                    acoustic = speech_result["acoustic_features"]
                else:
                    transcript = typed_answer or "[No answer provided]"
                    acoustic = SpeechProcessor.save_audio_bytes  # not used
                    from src.speech_processor import AcousticAnalyzer
                    acoustic = AcousticAnalyzer._dummy_features()

                feedback = engine.evaluate(
                    question=q,
                    transcript=transcript,
                    acoustic_features=acoustic,
                )

            st.session_state.results.append(feedback)

            # Show quick result
            c_score = feedback["content_feedback"]["score"]
            conf_score = feedback["confidence_feedback"]["confidence_score"]
            overall = feedback["overall_score"]

            st.success("✅ Answer evaluated!")
            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Content Score", f"{c_score}/100")
            with r2:
                st.metric("Confidence", f"{conf_score}/100")
            with r3:
                st.metric("Overall", f"{overall}/100")

            with st.expander("📝 Quick Feedback"):
                cf = feedback["content_feedback"]
                st.markdown(f"**Comment:** {cf.get('overall_comment','')}")
                if cf.get("strengths"):
                    st.markdown("**Strengths:**")
                    for s in cf["strengths"]:
                        st.markdown(f"  ✅ {s}")
                if cf.get("improvements"):
                    st.markdown("**To Improve:**")
                    for i in cf["improvements"]:
                        st.markdown(f"  📌 {i}")

            time.sleep(1.5)
            if idx + 1 < total:
                st.session_state.current_q += 1
                st.rerun()
            else:
                st.session_state.stage = "report"
                st.rerun()


# ── Stage: REPORT ──────────────────────────────────────────────────────────────
elif st.session_state.stage == "report":
    results = st.session_state.results

    st.title("📊 Interview Report")
    if not results:
        st.warning("No answers were recorded. Please restart and attempt at least one question.")
        st.stop()

    # ── Overall summary ───────────────────────────────────────────────────────
    avg_content    = sum(r["content_feedback"]["score"] for r in results) / len(results)
    avg_confidence = sum(r["confidence_feedback"]["confidence_score"] for r in results) / len(results)
    avg_overall    = sum(r["overall_score"] for r in results) / len(results)

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(
            f'<div class="score-card {score_color(int(avg_overall))}">'
            f'<h2>{avg_overall:.0f}</h2><p>Overall Score</p></div>',
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f'<div class="score-card {score_color(int(avg_content))}">'
            f'<h2>{avg_content:.0f}</h2><p>Content Quality</p></div>',
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            f'<div class="score-card {score_color(int(avg_confidence))}">'
            f'<h2>{avg_confidence:.0f}</h2><p>Confidence</p></div>',
            unsafe_allow_html=True,
        )
    with r4:
        st.markdown(
            f'<div class="score-card score-medium">'
            f'<h2>{len(results)}/{len(st.session_state.questions)}</h2><p>Questions Answered</p></div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.subheader("📈 Performance Overview")

    col_radar, col_bar = st.columns(2)

    with col_radar:
        if results:
            breakdown_avg = {}
            for r in results:
                for k, v in r["confidence_feedback"]["breakdown"].items():
                    breakdown_avg[k] = breakdown_avg.get(k, 0) + v
            for k in breakdown_avg:
                breakdown_avg[k] = round(breakdown_avg[k] / len(results))

            categories = list(breakdown_avg.keys())
            values = list(breakdown_avg.values())
            categories_display = [c.replace("_", " ").title() for c in categories]

            fig_radar = go.Figure(go.Scatterpolar(
                r=values + [values[0]],
                theta=categories_display + [categories_display[0]],
                fill="toself",
                fillcolor="rgba(102,126,234,0.25)",
                line=dict(color="#667eea", width=2),
                name="Confidence Breakdown",
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=False,
                title="Confidence Dimensions",
                height=340,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    with col_bar:
        df_scores = pd.DataFrame([
            {
                "Question": f"Q{r['question_id']}",
                "Content": r["content_feedback"]["score"],
                "Confidence": r["confidence_feedback"]["confidence_score"],
                "Overall": r["overall_score"],
            }
            for r in results
        ])
        fig_bar = px.bar(
            df_scores.melt(id_vars="Question", var_name="Metric", value_name="Score"),
            x="Question", y="Score", color="Metric",
            barmode="group",
            color_discrete_map={"Content": "#667eea", "Confidence": "#27ae60", "Overall": "#f39c12"},
            title="Score Breakdown per Question",
            height=340,
        )
        fig_bar.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Per-question details ───────────────────────────────────────────────────
    st.subheader("📋 Detailed Question Feedback")

    for r in results:
        cf = r["content_feedback"]
        conff = r["confidence_feedback"]

        with st.expander(
            f"Q{r['question_id']} — {r['question_text'][:80]}... | "
            f"Overall: {r['overall_score']}/100",
            expanded=False,
        ):
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**🗣️ Your Transcript:**")
                st.info(r["transcript"] or "_No transcript_")
                st.markdown("**✅ Strengths:**")
                for s in cf.get("strengths", []):
                    st.markdown(f"- {s}")
                st.markdown("**📌 Improvements:**")
                for imp in cf.get("improvements", []):
                    st.markdown(f"- {imp}")

            with col_b:
                st.markdown("**📊 Confidence Breakdown:**")
                for dim, val in conff["breakdown"].items():
                    label = dim.replace("_", " ").title()
                    color = "green" if val >= 70 else "orange" if val >= 50 else "red"
                    st.markdown(f"**{label}:** :{color}[{val}/100]")
                    st.progress(val / 100)

                st.markdown("**💡 Confidence Tips:**")
                for tip in conff.get("tips", []):
                    st.markdown(f"- {tip}")

    # ── Download report ───────────────────────────────────────────────────────
    st.divider()
    report_data = {
        "summary": {
            "avg_content": round(avg_content, 1),
            "avg_confidence": round(avg_confidence, 1),
            "avg_overall": round(avg_overall, 1),
            "questions_attempted": len(results),
        },
        "results": results,
    }
    report_json = json.dumps(report_data, indent=2, default=str)

    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "⬇️ Download JSON Report",
            data=report_json,
            file_name="interview_report.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_dl2:
        if st.button("🔄 Start New Interview", type="primary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

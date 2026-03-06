# 🎙️ AI Mock Interviewer

An end-to-end AI-powered interview preparation tool. Upload any job description, answer AI-generated role-specific questions via audio or text, and receive detailed feedback on **content quality** and **confidence score**.

---

## 🚀 Features

| Feature | Details |
|---|---|
| **Smart Question Generation** | GPT-4o-mini / Llama3 / offline mock mode |
| **Speech-to-Text** | OpenAI Whisper (local or API) |
| **Content Feedback** | Relevance, keyword hits, STAR structure, grade A–F |
| **Confidence Score** | Acoustic analysis (pitch, pause ratio, energy) + linguistic cues |
| **Interactive Dashboard** | Radar chart, per-question drilldown, downloadable JSON report |
| **PDF JD Upload** | Extract text from uploaded job description PDFs |
| **Fully Offline Mode** | Mock backends work without any API key |

---

## 📸 App Flow

```
Job Description → Question Generation → [Record / Type Answer] → Transcription → Feedback → Report
```

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit + Plotly
- **Speech-to-Text**: OpenAI Whisper (local `openai-whisper` or API)
- **LLM**: OpenAI GPT-4o-mini OR Ollama (Llama 3, Mistral, etc.)
- **Acoustic Analysis**: librosa (pitch, energy, pause ratio)
- **Feedback Engine**: Rule-based (offline) or LLM-based

---

## ⚡ Quick Start

### Option 1 — Local Python

```bash
# Clone / navigate to this folder
cd ai-mock-interviewer

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Set API key for OpenAI features
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run the app
streamlit run streamlit_app.py
```

### Option 2 — Docker

```bash
cp .env.example .env  # add your API key
docker-compose up --build
# Open http://localhost:8501
```

---

## ⚙️ Backend Options

### Question Generator
| Backend | How to activate | Requirement |
|---|---|---|
| `mock` | Default | None |
| `openai` | Set `OPENAI_API_KEY` | OpenAI API key |
| `ollama` | Run Ollama locally | `ollama pull llama3` |

### Speech-to-Text
| Backend | How to activate | Requirement |
|---|---|---|
| `mock` | Default | None |
| `local_whisper` | Select in sidebar | `pip install openai-whisper` + ffmpeg |
| `openai_whisper` | Set `OPENAI_API_KEY` | OpenAI API key |

---

## 📁 Project Structure

```
ai-mock-interviewer/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── src/
    ├── __init__.py
    ├── question_generator.py  # GPT / Ollama / Mock question generation
    ├── speech_processor.py    # Whisper transcription + acoustic analysis
    └── feedback_engine.py     # Content scoring + confidence scoring
```

---

## 🎯 Why This Project Stands Out

1. **Solves a real student problem** — interview preparation is ubiquitous
2. **Multi-modal AI pipeline** — LLM + ASR + acoustic analysis in one app
3. **Fully demo-able** — works without any API key in mock mode
4. **Modular** — swap out any component (GPT → Llama, Whisper → Deepgram)
5. **Recruiter-friendly** — shows NLP, audio ML, LLM integration, full-stack skills

---

## 🔮 Future Enhancements

- WebRTC in-browser audio recording (no file upload needed)
- Video analysis (eye contact, facial expression confidence)
- Multi-language support
- Interview history database (SQLite / Supabase)
- Leaderboard / progress tracking over time

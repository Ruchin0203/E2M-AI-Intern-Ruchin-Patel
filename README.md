# Resume–JD Matching Tool (with AI Scoring)

An AI-assisted recruiter tool that compares a candidate's resume (PDF) against
a pasted job description, and produces an **explainable match score**, a
skill-level breakdown, and a recruiter-style narrative explanation.

---

## Why this isn't "just a keyword matcher"

- **LLM (Gemini via LangChain)** is used *only* to extract and structure
  information from the resume/JD, and to write the final narrative
  explanation. It never calculates scores or decides matches.
- **All scoring, matching, and decision logic lives in plain, deterministic
  Python** (`matcher.py`, `scorer.py`, `decision_engine.py`) — exact match,
  fuzzy match (RapidFuzz), semantic similarity (Sentence-Transformers cosine
  similarity), transferable-skill detection, project relevance, and
  experience matching.
- If no Gemini API key is configured, the backend **automatically falls back**
  to a deterministic keyword/NLP-based extractor so the project is always
  runnable end-to-end, even with zero API cost.

---

## Project Structure

```
resume-jd-matcher/
├── backend/
│   ├── app.py                 # FastAPI entrypoint, wires the full pipeline
│   ├── config.py               # Settings, env vars, logging
│   ├── parser.py                # PDF text extraction (PyMuPDF) — no AI
│   ├── resume_analyzer.py       # Resume -> structured JSON (LangChain + fallback)
│   ├── jd_analyzer.py           # JD -> structured JSON (LangChain + fallback)
│   ├── matcher.py               # Deterministic matching engine (no AI)
│   ├── scorer.py                # Deterministic scoring engine (no AI)
│   ├── decision_engine.py       # Deterministic recruiter decision logic (no AI)
│   ├── llm_service.py           # LCEL chains: extraction + explanation
│   ├── prompt_templates.py      # All LLM prompts, kept out of business logic
│   ├── transferable_skills.py   # Skill category taxonomy
│   ├── models.py                # Pydantic schemas
│   ├── utils.py                 # Shared helpers
│   ├── requirements.txt
│   └── .env.example
└── frontend/
    ├── src/
    │   ├── components/          # UploadResume, ScoreCard, ResumeViewer, etc.
    │   ├── pages/Dashboard.jsx   # Orchestrates the whole UI flow
    │   ├── services/api.js      # Axios client -> /analyze
    │   ├── utils/highlighter.js # Builds highlighted resume/JD HTML
    │   └── App.jsx
    ├── package.json
    └── .env.example
```

---

## Architecture / Data Flow

```
Resume PDF ──► Parser (PyMuPDF) ──► Resume Analyzer (LangChain/Gemini) ──► Candidate Profile JSON
                                                                                    │
Job Description ──► JD Analyzer (LangChain/Gemini) ──► Requirement Profile JSON ──┤
                                                                                    ▼
                                                              Matching Engine (pure Python)
                                                                          │
                                                              Scoring Engine (pure Python)
                                                                          │
                                                              Decision Engine (pure Python)
                                                                          │
                                                     Explanation Chain (LangChain/Gemini, narrative only)
                                                                          │
                                                                          ▼
                                                                     JSON Response ──► React Frontend
```

**LangChain is used only for orchestration of LLM calls** (LCEL: `PromptTemplate | ChatGoogleGenerativeAI | JsonOutputParser`). No agents, no memory, no LangGraph, no RetrievalQA — this project doesn't need an agentic workflow.

### Scoring weights (fixed, documented, explainable)

| Component            | Weight |
|-----------------------|-------|
| Exact Skill Match      | 40%   |
| Semantic Similarity    | 20%   |
| Transferable Skills    | 15%   |
| Project Relevance      | 15%   |
| Experience             | 5%    |
| Education              | 5%    |

Each component returns a `raw_score`, `weighted_score`, and `confidence` so the final number is always explainable, never a black box.

---

## Installation & Running

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt --break-system-packages   # or drop the flag in a venv

cp .env.example .env
# Optionally add your Gemini key to .env:
# GOOGLE_API_KEY=your_key_here
# Get a free key at https://aistudio.google.com/apikey

uvicorn app:app --reload
```

The backend runs at `http://localhost:8000`. Interactive API docs are auto-generated at `http://localhost:8000/docs`.

> **No Gemini key?** The app still runs. It automatically uses a deterministic
> keyword/regex-based extractor instead of Gemini for resume/JD parsing and a
> template-based explanation, so you can demo the entire pipeline for free.

### 2. Frontend

```bash
cd frontend
npm install
cp .env.example .env   # defaults to http://localhost:8000, edit if needed
npm run dev
```

The frontend runs at `http://localhost:5173`.

### 3. Use it

1. Open `http://localhost:5173`
2. Upload a resume PDF
3. Paste a job description
4. Click **Analyze Match**
5. Review the score, skill breakdown, highlighted resume/JD, and recruiter explanation

---

## Environment Variables

**Backend (`backend/.env`)**

| Variable | Description | Default |
|---|---|---|
| `GOOGLE_API_KEY` | Gemini API key. Leave empty to use the deterministic fallback. | *(empty)* |
| `GEMINI_MODEL` | Gemini model name | `gemini-1.5-flash` |
| `LLM_TEMPERATURE` | LLM sampling temperature | `0.2` |
| `FUZZY_MATCH_THRESHOLD` | RapidFuzz match threshold (0-100) | `85` |
| `SEMANTIC_MATCH_THRESHOLD` | Cosine similarity threshold (0-1) | `0.62` |
| `SENTENCE_TRANSFORMER_MODEL` | Embedding model | `all-MiniLM-L6-v2` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `http://localhost:5173` |

**Frontend (`frontend/.env`)**

| Variable | Description | Default |
|---|---|---|
| `VITE_API_BASE_URL` | Backend base URL | `http://localhost:8000` |

---

## API

### `POST /analyze`

Multipart form data:
- `resume`: PDF file
- `job_description`: string

Returns a JSON object including `overall_score`, `score_breakdown`, `matched_skills`, `missing_skills`, `transferable_skills`, `project_relevance`, `explanation`, and the raw resume/JD text (used by the frontend for highlighting).

### `GET /health`
Simple liveness check.

---

## Features

- PDF resume parsing (PyMuPDF), pasted job description input
- LLM-powered structured extraction of resume & JD (Gemini via LangChain LCEL)
- Deterministic exact, fuzzy (RapidFuzz), and semantic (Sentence-Transformers) skill matching
- Transferable-skill detection via a category taxonomy (e.g., Flask ↔ FastAPI)
- Project-relevance scoring against JD responsibilities/skills
- Fresher-friendly experience matching (doesn't over-penalize limited years when projects are highly relevant)
- Fully explainable, weighted scoring engine with per-component confidence
- Deterministic recruiter-style decision engine (Excellent/Strong/Good/Potential/Weak Match)
- LLM-generated recruiter narrative explanation, with a deterministic fallback if no API key is set
- React + Tailwind dashboard: score gauge, breakdown bars, matched/missing/transferable chips, highlighted resume & JD viewers, recruiter explanation card
- Graceful error handling and retry on the frontend; input validation on the backend

## Known Limitations

- The deterministic fallback extractor (used when no Gemini key is set) relies on keyword/regex matching and is less nuanced than the LLM-based extraction.
- Semantic matching and project relevance require `sentence-transformers` (and its `torch` dependency), which increases install size/time.
- PDF parsing assumes a text layer is present; scanned/image-only resumes are not OCR'd.
- No authentication, database, or persistence layer — this is a stateless, single-request tool by design (per assessment scope).

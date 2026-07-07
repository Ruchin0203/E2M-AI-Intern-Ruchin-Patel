# Resume–JD Matching Tool

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-8E75B2?style=for-the-badge&logo=googlegemini&logoColor=white)
![React](https://img.shields.io/badge/React-Vite-61DAFB?style=for-the-badge&logo=react&logoColor=black)

**AI-assisted recruiter tool that scores a resume against a job description — explainably.**
**No black-box LLM scoring. Every number is traceable back to deterministic Python logic.**

[**How It Works**](#how-it-works) · [**Quick Start**](#quick-start) · [**Scoring Model**](#scoring-model) · [**Screenshots**](#screenshots) · [**Author**](#author)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [How It Works](#how-it-works)
  - [Extraction vs Business Logic](#extraction-vs-business-logic)
  - [Matching Engine](#matching-engine)
  - [Scoring Model](#scoring-model)
  - [Decision Engine](#decision-engine)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Screenshots](#screenshots)
- [Known Limitations](#known-limitations)
- [Author](#author)

---

## Overview

Most resume screeners are glorified **keyword matchers** — they count overlapping words and call it a score. That approach can't tell the difference between a candidate who *actually* built with FastAPI and one who merely listed it once, and it can't recognize that "Flask experience" is still a meaningful signal for a "FastAPI required" role.

This project imitates how an experienced recruiter actually evaluates a candidate:

- Reads the resume and job description **structurally**, not just as bags of words
- Distinguishes **exact matches**, **fuzzy matches** (typos/naming variants), **semantic matches** (`LLM` ≈ `Large Language Model`), and **transferable skills** (`Flask` → `Backend Framework` ← `FastAPI`)
- Weighs **project relevance** and **experience** the way a recruiter would — a fresher with a highly relevant project isn't penalized as heavily as raw "years of experience" would suggest
- Produces a **fully explainable score**, broken into weighted components, each with its own confidence
- Uses an LLM **only** to read and summarize — never to decide the number

---

## Key Features

- **PDF Resume Parsing** — clean text extraction via PyMuPDF, no OCR dependency for text-based PDFs
- **LLM-Powered Structured Extraction** — Gemini via LangChain (LCEL) converts unstructured resume/JD text into structured JSON
- **Deterministic Matching Engine** — exact match, RapidFuzz fuzzy match, Sentence-Transformer semantic similarity, all pure Python, zero LLM calls
- **Transferable Skill Taxonomy** — category-based mapping (Cloud, Backend Framework, Deep Learning, Vector DB, etc.) catches adjacent skills a keyword matcher would miss
- **Project Relevance Scoring** — semantic similarity between candidate projects and JD responsibilities, labeled High / Medium / Low
- **Fresher-Friendly Experience Matching** — doesn't over-penalize limited years when project relevance is high
- **Explainable Weighted Scoring** — every component reports raw score, weighted score, and confidence
- **Deterministic Decision Engine** — Excellent / Strong / Good / Potential / Weak Match, with strengths, weaknesses, and a recommendation
- **Recruiter-Style Narrative Explanation** — LLM-generated summary that explains the *already-calculated* score, never recalculates it
- **Automatic Offline Fallback** — if no Gemini API key is configured, the entire pipeline still runs end-to-end using a deterministic keyword/NLP extractor
- **Resume & JD Highlighting** — matched skills in green, missing in red, transferable in yellow, directly in the original document text
- **Zero infrastructure** — no database, no auth, no Docker required to run locally

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     config.py (settings, weights)                 │
└───────────────┬────────────────┬─────────────────┬───────────────┘
                │                │                 │
        ┌───────▼───────┐ ┌──────▼───────┐ ┌───────▼────────────┐
        │  parser.py     │ │ resume_      │ │  jd_analyzer.py     │
        │  (PyMuPDF)     │ │ analyzer.py  │ │                     │
        │  No AI         │ │ LangChain +  │ │ LangChain +         │
        │                │ │ fallback NLP │ │ fallback NLP        │
        └────────────────┘ └──────┬───────┘ └──────────┬──────────┘
                                   │                    │
                                   ▼                    ▼
                          Candidate Profile JSON   Requirement Profile JSON
                                   │                    │
                                   └─────────┬──────────┘
                                             ▼
                              ┌───────────────────────────┐
                              │       matcher.py           │
                              │  Exact / Fuzzy / Semantic  │
                              │  Transferable / Project /  │
                              │  Experience — pure Python  │
                              └──────────────┬─────────────┘
                                             ▼
                              ┌───────────────────────────┐
                              │       scorer.py            │
                              │  Weighted, explainable      │
                              │  score — pure Python         │
                              └──────────────┬─────────────┘
                                             ▼
                              ┌───────────────────────────┐
                              │   decision_engine.py        │
                              │  Recruiter decision rules   │
                              │  — pure Python               │
                              └──────────────┬─────────────┘
                                             ▼
                              ┌───────────────────────────┐
                              │   llm_service.py            │
                              │  Explanation Chain          │
                              │  (narrative only, LangChain)│
                              └──────────────┬─────────────┘
                                             ▼
                                     JSON Response
                                             │
                                             ▼
                                  React + Tailwind Frontend
```

**Module responsibilities:**

| File | Responsibility |
|------|-----------------|
| `app.py` | FastAPI entrypoint — wires the full pipeline, exposes `/analyze` |
| `config.py` | All tunable settings (scoring weights, thresholds, model names) in one place |
| `parser.py` | PDF → clean text (PyMuPDF). No AI, no business logic |
| `resume_analyzer.py` | Resume text → structured `ResumeProfile` JSON (LangChain, with deterministic fallback) |
| `jd_analyzer.py` | JD text → structured `JDProfile` JSON (LangChain, with deterministic fallback) |
| `matcher.py` | Deterministic exact/fuzzy/semantic/transferable/project/experience matching |
| `scorer.py` | Deterministic, weighted, explainable scoring engine |
| `decision_engine.py` | Deterministic recruiter-style decision rules |
| `llm_service.py` | LCEL chains for extraction and narrative explanation only |
| `prompt_templates.py` | All LLM prompts, kept out of business logic |
| `transferable_skills.py` | Skill category taxonomy (Cloud, Backend, Deep Learning, etc.) |
| `models.py` | Pydantic schemas for every stage of the pipeline |

---

## How It Works

### Extraction vs Business Logic

The single most important design decision in this project: **the LLM never scores anything.**

| Responsibility | Owner |
|---|---|
| Reading & structuring resume/JD text | LangChain + Gemini |
| Writing the final narrative explanation | LangChain + Gemini |
| Matching skills (exact/fuzzy/semantic) | Pure Python |
| Calculating the score | Pure Python |
| Deciding the recommendation | Pure Python |

This means the score is **reproducible** — run the same resume against the same JD twice, and (parsing aside) you get the same number, because the math is deterministic Python, not a temperature-sampled LLM call.

### Matching Engine

Skills are matched in four escalating passes, each one only considering what the previous pass missed:

1. **Exact match** — case-insensitive direct string match (`Python` = `Python`)
2. **Fuzzy match** — RapidFuzz token-sort ratio catches naming variants (`Fast API` ≈ `FastAPI`)
3. **Semantic match** — Sentence-Transformer cosine similarity catches conceptual equivalence (`REST API` ≈ `RESTful API`, `LLM` ≈ `Large Language Model`)
4. **Transferable skill match** — category taxonomy catches adjacent skills (`Flask` → *Backend Framework* ← `FastAPI`)

Anything left after all four passes is reported as a genuinely **missing** skill.

### Scoring Model

| Component | Weight | What it measures |
|---|---|---|
| Exact Skill Match | 40% | Direct skill overlap |
| Semantic Similarity | 20% | Fuzzy + semantic matches |
| Transferable Skills | 15% | Adjacent/category-based matches |
| Project Relevance | 15% | Similarity of projects to JD responsibilities |
| Experience | 5% | Years vs. required, adjusted for strong projects |
| Education | 5% | Degree/field alignment |

Each component returns a `raw_score`, `weighted_score`, and `confidence` — so the final `overall_score` is always traceable back to *why* it landed where it did, never a single opaque number.

### Decision Engine

The weighted `overall_score` maps to a recruiter-style category:

| Score | Decision |
|---|---|
| 90+ | Excellent Match |
| 80–89 | Strong Match |
| 70–79 | Good Match |
| 55–69 | Potential Match |
| < 55 | Weak Match |

Strengths, weaknesses, missing skills, and a recommendation are derived deterministically from the score breakdown and matching data — the LLM's job at this stage is purely to phrase it well, not to decide it.

---

## Project Structure

```
resume-jd-matcher/
│
├── backend/
│   ├── app.py                 # FastAPI entrypoint
│   ├── config.py               # Settings, weights, thresholds
│   ├── parser.py                # PDF text extraction — no AI
│   ├── resume_analyzer.py       # Resume → structured JSON
│   ├── jd_analyzer.py           # JD → structured JSON
│   ├── matcher.py               # Deterministic matching engine
│   ├── scorer.py                # Deterministic scoring engine
│   ├── decision_engine.py       # Deterministic decision logic
│   ├── llm_service.py           # LCEL chains (extraction + explanation)
│   ├── prompt_templates.py      # All LLM prompts
│   ├── transferable_skills.py   # Skill category taxonomy
│   ├── models.py                # Pydantic schemas
│   ├── utils.py                 # Shared helpers
│   ├── requirements.txt
│   └── .env.example
│
└── frontend/
    ├── src/
    │   ├── components/          # UploadResume, ScoreCard, ResumeViewer, etc.
    │   ├── pages/Dashboard.jsx   # Orchestrates the full UI flow
    │   ├── services/api.js      # Axios client → /analyze
    │   ├── utils/highlighter.js # Builds highlighted resume/JD HTML
    │   └── App.jsx
    ├── package.json
    └── .env.example
```

---

## Quick Start

### Prerequisites

- Python **3.11** (recommended — some ML packages lack Windows wheels for 3.13)
- Node.js **18+**
- No GPU required — runs entirely on CPU

### Backend Setup

```bash
cd backend
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

cp .env.example .env
# Optionally add a Gemini key to .env:
# GOOGLE_API_KEY=your_key_here   (free at https://aistudio.google.com/apikey)

uvicorn app:app --reload
```

Backend runs at `http://localhost:8000`. Interactive API docs: `http://localhost:8000/docs`.

> **No Gemini key?** The app still runs completely — it automatically switches to a deterministic keyword/NLP extractor for resume/JD parsing and a template-based explanation, so the full pipeline works at zero API cost.

### Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env
npm run dev
```

Frontend runs at `http://localhost:5173`. Upload a resume PDF, paste a job description, and click **Analyze Match**.

---

## Configuration

All scoring weights and thresholds live in `backend/config.py` — no other file needs editing for basic tuning:

```python
# ── Scoring Weights (must sum to 100) ─────────────────────────────
WEIGHT_EXACT_SKILLS   = 40
WEIGHT_SEMANTIC       = 20
WEIGHT_TRANSFERABLE   = 15
WEIGHT_PROJECT        = 15
WEIGHT_EXPERIENCE     = 5
WEIGHT_EDUCATION      = 5

# ── Matching Thresholds ────────────────────────────────────────────
FUZZY_MATCH_THRESHOLD     = 85     # RapidFuzz score (0–100)
SEMANTIC_MATCH_THRESHOLD  = 0.62   # Cosine similarity (0–1)
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# ── LLM ─────────────────────────────────────────────────────────────
GEMINI_MODEL       = "gemini-1.5-flash"
LLM_TEMPERATURE    = 0.2
```

---

## API Reference

### `POST /analyze`

Multipart form data:

| Field | Type | Description |
|---|---|---|
| `resume` | file | Candidate resume, PDF only |
| `job_description` | string | Pasted job description text |

Returns `overall_score`, `score_breakdown`, `matched_skills`, `missing_skills`, `transferable_skills`, `project_relevance`, `explanation`, and the raw resume/JD text used for frontend highlighting.

### `GET /health`

Simple liveness check — returns `{"status": "ok"}`.

---

## Screenshots

> Add your own screenshots here after running the app locally, e.g.:
> `![Dashboard](docs/dashboard.png)`

| View | Description |
|---|---|
| Score Card | Circular gauge + summary banner with matched/missing counts |
| Score Breakdown | Per-component weighted bars with raw percentages |
| Resume/JD Viewer | Original text with matched (green), missing (red), transferable (yellow) highlights |
| Explanation Card | Recruiter-style narrative: strengths, weaknesses, suggestions, recommendation |

---

## Known Limitations

- The offline fallback extractor (used when no Gemini key is set) relies on keyword/regex matching and is less nuanced than LLM-based extraction
- Semantic matching and project relevance require `sentence-transformers` (and its `torch` dependency), which increases install size and first-run load time
- PDF parsing assumes a text layer is present — scanned/image-only resumes are not OCR'd
- No authentication, database, or persistence layer — this is intentionally a stateless, single-request tool

---

## Author

<table>
  <tr>
    <td><strong>Developer</strong></td>
    <td>Ruchinkumar Hiteshbhai Patel</td>
  </tr>
  <tr>
    <td><strong>Degree</strong></td>
    <td>B.Tech Computer Science &amp; Engineering (AI-ML)</td>
  </tr>
  <tr>
    <td><strong>Institution</strong></td>
    <td>Adani University, Ahmedabad, Gujarat, India</td>
  </tr>
  <tr>
    <td><strong>GitHub</strong></td>
    <td><a href="https://github.com/Ruchin0203">@Ruchin0203</a></td>
  </tr>
</table>

---

## License

This project is released under the [MIT License](LICENSE).

---

<div align="center">

Made with ❤️ · B.Tech CSE (AI-ML) 2025–26

⭐ If this project helped you, please consider giving it a star!

</div>

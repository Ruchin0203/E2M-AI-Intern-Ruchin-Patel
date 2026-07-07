"""
resume_analyzer.py
Converts unstructured resume text into a structured ResumeProfile.

Primary path: LangChain + Gemini (via llm_service.extract_resume_profile).
Fallback path: deterministic keyword/regex based NLP extraction, used when
no GOOGLE_API_KEY is configured or the LLM call fails, so the project is
always runnable end-to-end.
"""

import re
from typing import List

from pydantic import ValidationError

from models import ResumeProfile
from llm_service import extract_resume_profile as llm_extract_resume_profile
from llm_service import LLMNotConfiguredError, LLMExtractionError
from utils import dedupe_preserve_order, extract_years_from_text
from config import get_logger

logger = get_logger(__name__)

# --- Keyword taxonomy used only by the deterministic fallback extractor -----

_LANGUAGES = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "kotlin", "swift", "php", "ruby", "r", "sql", "html", "css", "dart",
]
_FRAMEWORKS = [
    "fastapi", "flask", "django", "react", "react.js", "vue", "vue.js",
    "angular", "next.js", "nextjs", "express", "express.js", "spring",
    "spring boot", "nestjs", "laravel", "node.js", "nodejs",
]
_LIBRARIES = [
    "langchain", "langgraph", "numpy", "pandas", "scikit-learn", "sklearn",
    "opencv", "matplotlib", "seaborn", "transformers", "huggingface",
    "tensorflow", "pytorch", "keras", "yolov8", "yolov5", "rapidfuzz",
    "spacy", "sentence-transformers", "beautifulsoup", "selenium",
]
_CLOUD = ["aws", "azure", "gcp", "google cloud", "digitalocean", "heroku", "oracle cloud"]
_DATABASES = [
    "mysql", "postgresql", "postgres", "mongodb", "sqlite", "redis",
    "faiss", "pinecone", "chromadb", "dynamodb", "cassandra",
]
_TOOLS = [
    "git", "github", "docker", "kubernetes", "n8n", "postman", "jira",
    "linux", "vscode", "jupyter", "figma",
]
_SOFT_SKILLS = [
    "communication", "teamwork", "leadership", "problem solving",
    "adaptability", "collaboration", "time management", "critical thinking",
]
_DOMAINS = [
    "computer vision", "natural language processing", "nlp",
    "machine learning", "deep learning", "generative ai", "genai",
    "retrieval augmented generation", "rag", "llm", "large language model",
    "traffic management", "recommendation systems", "data engineering",
]

_ALL_SKILL_GROUPS = {
    "skills": _LANGUAGES + _DOMAINS,
    "frameworks": _FRAMEWORKS,
    "libraries": _LIBRARIES,
    "cloud": _CLOUD,
    "databases": _DATABASES,
    "tools": _TOOLS,
    "soft_skills": _SOFT_SKILLS,
    "domains": _DOMAINS,
}


def _find_keywords(text: str, keywords: List[str]) -> List[str]:
    lowered = text.lower()
    found = []
    for kw in keywords:
        pattern = r"(?<![a-z0-9])" + re.escape(kw.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, lowered):
            found.append(kw)
    return found


def _extract_section(text: str, headers: List[str]) -> List[str]:
    """Best-effort extraction of lines under a resume section header."""
    lines = text.split("\n")
    collected = []
    capturing = False
    header_pattern = re.compile(
        r"^\s*(" + "|".join(re.escape(h) for h in headers) + r")\s*:?\s*$", re.IGNORECASE
    )
    stop_pattern = re.compile(
        r"^\s*(skills|projects|experience|education|certifications|achievements|"
        r"internship|summary|objective|contact)\s*:?\s*$",
        re.IGNORECASE,
    )
    for line in lines:
        if header_pattern.match(line):
            capturing = True
            continue
        if capturing and stop_pattern.match(line) and not header_pattern.match(line):
            break
        if capturing and line.strip():
            collected.append(line.strip())
    return collected[:15]


def _fallback_extract(resume_text: str) -> dict:
    """Deterministic rule-based extraction used when the LLM is unavailable."""
    logger.info("Using deterministic fallback extractor for resume analysis.")

    profile = {group: _find_keywords(resume_text, kws) for group, kws in _ALL_SKILL_GROUPS.items()}

    profile["projects"] = _extract_section(resume_text, ["projects", "project", "academic projects"])
    profile["experience"] = _extract_section(resume_text, ["experience", "work experience", "professional experience"])
    profile["education"] = _extract_section(resume_text, ["education", "academics", "academic background"])
    profile["certifications"] = _extract_section(resume_text, ["certifications", "certificates", "licenses"])
    profile["internship_experience"] = _extract_section(resume_text, ["internship", "internships"])
    profile["achievements"] = _extract_section(resume_text, ["achievements", "awards", "accomplishments"])
    profile["total_experience_years"] = extract_years_from_text(resume_text)

    for key in ("skills", "frameworks", "libraries", "cloud", "databases", "tools", "soft_skills", "domains"):
        profile[key] = dedupe_preserve_order(profile.get(key, []))

    return profile


def analyze_resume(resume_text: str) -> ResumeProfile:
    """
    Analyze resume text and return a validated ResumeProfile.
    Tries the LLM extraction chain first, falling back to deterministic
    NLP extraction if the LLM is not configured or the call fails.
    """
    raw_profile = None
    try:
        raw_profile = llm_extract_resume_profile(resume_text)
    except LLMNotConfiguredError:
        logger.warning("LLM not configured; falling back to rule-based resume extraction.")
    except LLMExtractionError as exc:
        logger.error("LLM resume extraction failed, using fallback: %s", exc)

    if raw_profile is None:
        raw_profile = _fallback_extract(resume_text)

    raw_profile["raw_text"] = resume_text

    try:
        return ResumeProfile(**raw_profile)
    except ValidationError as exc:
        logger.error("Resume profile validation failed, using safe fallback: %s", exc)
        fallback = _fallback_extract(resume_text)
        fallback["raw_text"] = resume_text
        return ResumeProfile(**fallback)

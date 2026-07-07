"""
jd_analyzer.py
Converts an unstructured job description into a structured JDProfile.

Primary path: LangChain + Gemini (via llm_service.extract_job_profile).
Fallback path: deterministic keyword/regex based NLP extraction.
"""

import re
from typing import List

from pydantic import ValidationError

from models import JDProfile
from llm_service import extract_job_profile as llm_extract_job_profile
from llm_service import LLMNotConfiguredError, LLMExtractionError
from resume_analyzer import _LANGUAGES, _FRAMEWORKS, _CLOUD, _DATABASES, _SOFT_SKILLS, _DOMAINS, _find_keywords
from utils import dedupe_preserve_order, extract_years_from_text
from config import get_logger

logger = get_logger(__name__)

_MUST_HAVE_MARKERS = [
    "must have", "required", "requirements", "mandatory", "essential", "you have",
]
_NICE_TO_HAVE_MARKERS = [
    "nice to have", "preferred", "good to have", "bonus", "plus point", "added advantage",
]


def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.\n])\s+", text) if s.strip()]


def _classify_requirement_sentences(jd_text: str):
    must_have, nice_to_have, responsibilities = [], [], []
    for sentence in _split_sentences(jd_text):
        lowered = sentence.lower()
        if any(marker in lowered for marker in _NICE_TO_HAVE_MARKERS):
            nice_to_have.append(sentence)
        elif any(marker in lowered for marker in _MUST_HAVE_MARKERS):
            must_have.append(sentence)
        elif re.match(r"^\s*[-•*]", sentence) or "responsib" in lowered:
            responsibilities.append(sentence)
    return must_have[:20], nice_to_have[:20], responsibilities[:20]


def _fallback_extract(jd_text: str) -> dict:
    """Deterministic rule-based extraction used when the LLM is unavailable."""
    logger.info("Using deterministic fallback extractor for JD analysis.")

    languages = _find_keywords(jd_text, _LANGUAGES)
    frameworks = _find_keywords(jd_text, _FRAMEWORKS)
    cloud = _find_keywords(jd_text, _CLOUD)
    databases = _find_keywords(jd_text, _DATABASES)
    soft_skills = _find_keywords(jd_text, _SOFT_SKILLS)
    ai_tech = _find_keywords(jd_text, _DOMAINS)

    must_have, nice_to_have, responsibilities = _classify_requirement_sentences(jd_text)

    all_tech_skills = dedupe_preserve_order(languages + frameworks + cloud + databases + ai_tech)

    # Actual skill list used for matching always comes from detected
    # technical keywords (the "must_have"/"nice_to_have" sentence lists
    # above are used only for the responsibilities/context fields).
    education = []
    for kw in ["b.tech", "btech", "bachelor", "master", "m.tech", "b.e.", "computer science", "engineering"]:
        if kw in jd_text.lower():
            education.append(kw)

    return {
        "must_have_skills": all_tech_skills,
        "nice_to_have_skills": nice_to_have,
        "responsibilities": responsibilities,
        "programming_languages": dedupe_preserve_order(languages),
        "frameworks": dedupe_preserve_order(frameworks),
        "databases": dedupe_preserve_order(databases),
        "cloud": dedupe_preserve_order(cloud),
        "experience_required_years": extract_years_from_text(jd_text),
        "education": dedupe_preserve_order(education),
        "soft_skills": dedupe_preserve_order(soft_skills),
        "ai_technologies": dedupe_preserve_order(ai_tech),
    }


def analyze_job_description(jd_text: str) -> JDProfile:
    """
    Analyze job description text and return a validated JDProfile.
    Tries the LLM extraction chain first, falling back to deterministic
    NLP extraction if the LLM is not configured or the call fails.
    """
    raw_profile = None
    try:
        raw_profile = llm_extract_job_profile(jd_text)
    except LLMNotConfiguredError:
        logger.warning("LLM not configured; falling back to rule-based JD extraction.")
    except LLMExtractionError as exc:
        logger.error("LLM JD extraction failed, using fallback: %s", exc)

    if raw_profile is None:
        raw_profile = _fallback_extract(jd_text)

    raw_profile["raw_text"] = jd_text

    try:
        return JDProfile(**raw_profile)
    except ValidationError as exc:
        logger.error("JD profile validation failed, using safe fallback: %s", exc)
        fallback = _fallback_extract(jd_text)
        fallback["raw_text"] = jd_text
        return JDProfile(**fallback)

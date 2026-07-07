"""
llm_service.py
Initializes the Gemini LLM and exposes LCEL chains for:
 - Resume information extraction
 - Job description information extraction
 - Recruiter-style explanation generation

This module contains NO business logic (no scoring, no matching).
If no GOOGLE_API_KEY is configured, callers should catch LLMNotConfiguredError
and fall back to the deterministic extractors in resume_analyzer.py / jd_analyzer.py.
"""

import json
from typing import Any, Dict, Optional

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from config import settings, get_logger
from prompt_templates import get_resume_prompt, get_jd_prompt, get_explanation_prompt
from utils import safe_json_loads

logger = get_logger(__name__)


class LLMNotConfiguredError(Exception):
    """Raised when a Gemini API key has not been configured."""


class LLMExtractionError(Exception):
    """Raised when the LLM response could not be parsed/validated."""


def _get_llm():
    """Lazily construct the ChatGoogleGenerativeAI client."""
    if not settings.USE_LLM:
        raise LLMNotConfiguredError(
            "GOOGLE_API_KEY is not set. Configure it in .env to enable LLM extraction."
        )
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GOOGLE_API_KEY,
        temperature=settings.LLM_TEMPERATURE,
    )


def _build_json_chain(prompt):
    """Build an LCEL chain: prompt -> llm -> JSON parser, with a raw-text fallback parse."""
    llm = _get_llm()
    parser = JsonOutputParser()

    def _parse(ai_message) -> Dict[str, Any]:
        content = ai_message.content if hasattr(ai_message, "content") else str(ai_message)
        try:
            return parser.parse(content)
        except Exception:  # noqa: BLE001
            return safe_json_loads(content)

    chain = prompt | llm | RunnableLambda(_parse)
    return chain


def extract_resume_profile(resume_text: str) -> Dict[str, Any]:
    """
    Execute the Resume Extraction Chain: converts raw resume text into a
    structured JSON dictionary matching the ResumeProfile schema.
    """
    prompt = get_resume_prompt()
    chain = _build_json_chain(prompt)
    try:
        result = chain.invoke({"resume_text": resume_text[:15000]})
    except Exception as exc:  # noqa: BLE001
        logger.error("Resume extraction chain failed: %s", exc)
        raise LLMExtractionError(str(exc)) from exc
    if not isinstance(result, dict):
        raise LLMExtractionError("Resume extraction did not return a JSON object.")
    return result


def extract_job_profile(jd_text: str) -> Dict[str, Any]:
    """
    Execute the JD Extraction Chain: converts raw job description text into a
    structured JSON dictionary matching the JDProfile schema.
    """
    prompt = get_jd_prompt()
    chain = _build_json_chain(prompt)
    try:
        result = chain.invoke({"jd_text": jd_text[:15000]})
    except Exception as exc:  # noqa: BLE001
        logger.error("JD extraction chain failed: %s", exc)
        raise LLMExtractionError(str(exc)) from exc
    if not isinstance(result, dict):
        raise LLMExtractionError("JD extraction did not return a JSON object.")
    return result


def generate_explanation(
    candidate_profile: Dict[str, Any],
    job_requirements: Dict[str, Any],
    score_breakdown: Dict[str, Any],
    decision: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """
    Execute the Explanation Chain: generates recruiter-style narrative
    explanation text. Never calculates scores or compares skills itself.

    Returns None if the LLM is not configured (caller should build a
    deterministic fallback explanation).
    """
    if not settings.USE_LLM:
        return None

    prompt = get_explanation_prompt()
    chain = _build_json_chain(prompt)
    try:
        result = chain.invoke(
            {
                "candidate_profile": json.dumps(candidate_profile, default=str),
                "job_requirements": json.dumps(job_requirements, default=str),
                "score_breakdown": json.dumps(score_breakdown, default=str),
                "decision": json.dumps(decision, default=str),
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Explanation chain failed: %s", exc)
        return None

    if not isinstance(result, dict):
        return None
    return result

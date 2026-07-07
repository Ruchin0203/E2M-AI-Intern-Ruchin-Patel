"""
utils.py
Small reusable helper functions shared across modules.
"""

import re
import json
from typing import Any, List


def clean_text(text: str) -> str:
    """Remove blank lines, duplicate spaces, and unicode artifacts."""
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = text.encode("utf-8", "ignore").decode("utf-8", "ignore")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line != ""]
    return "\n".join(lines).strip()


def normalize_skill(skill: str) -> str:
    """Lowercase and strip a skill string for comparison purposes."""
    if not skill:
        return ""
    skill = skill.strip().lower()
    skill = re.sub(r"[^a-z0-9+./# -]", "", skill)
    skill = re.sub(r"\s+", " ", skill).strip()
    return skill


def dedupe_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicates (case-insensitive) while preserving first occurrence order."""
    seen = set()
    result = []
    for item in items:
        key = normalize_skill(item)
        if key and key not in seen:
            seen.add(key)
            result.append(item.strip())
    return result


def safe_json_loads(text: str) -> Any:
    """
    Attempt to parse JSON from an LLM response, stripping markdown code fences
    and other common formatting artifacts if present.
    """
    if not isinstance(text, str):
        return text
    cleaned = text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"```$", "", cleaned.strip())
    cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    raise ValueError("Could not parse JSON from LLM response")


def extract_years_from_text(text: str) -> float:
    """Best-effort extraction of a numeric years-of-experience figure from text."""
    if not text:
        return 0.0
    match = re.search(r"(\d+(?:\.\d+)?)\s*\+?\s*(years|yrs|year)", text, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return 0.0
    return 0.0


def round_half_up(value: float, digits: int = 1) -> float:
    """Round a float to the given number of digits."""
    return round(float(value) + 1e-9, digits)


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    """Clamp a numeric value within [low, high]."""
    return max(low, min(high, value))

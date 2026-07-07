"""
scorer.py
Deterministic Scoring Engine. Contains NO AI/LLM calls.

Calculates the final explainable match score from the matching engine's
output, using fixed, documented weights.
"""

from typing import Dict

from config import settings
from utils import clamp, round_half_up
from models import ScoreBreakdown, ScoreComponent


def _exact_component(match_data: Dict) -> ScoreComponent:
    required = match_data["required_skills"]
    matched = match_data["exact_matched"]
    raw = (len(matched) / len(required) * 100) if required else 100.0
    weighted = raw * (settings.WEIGHT_EXACT_SKILLS / 100)
    confidence = 100.0 if required else 50.0
    return ScoreComponent(
        raw_score=round_half_up(raw), weighted_score=round_half_up(weighted), confidence=confidence
    )


def _semantic_component(match_data: Dict) -> ScoreComponent:
    required = match_data["required_skills"]
    fuzzy_count = len(match_data["fuzzy_matches"])
    semantic_count = len(match_data["semantic_matches"])
    raw = ((fuzzy_count + semantic_count) / len(required) * 100) if required else 0.0
    raw = clamp(raw)
    weighted = raw * (settings.WEIGHT_SEMANTIC / 100)
    confidence = 85.0 if (fuzzy_count + semantic_count) > 0 else 60.0
    return ScoreComponent(
        raw_score=round_half_up(raw), weighted_score=round_half_up(weighted), confidence=confidence
    )


def _transferable_component(match_data: Dict) -> ScoreComponent:
    required = match_data["required_skills"]
    transferable_count = len(match_data["transferable_skills"])
    raw = (transferable_count / len(required) * 100) if required else 0.0
    raw = clamp(raw)
    weighted = raw * (settings.WEIGHT_TRANSFERABLE / 100)
    confidence = 75.0 if transferable_count > 0 else 60.0
    return ScoreComponent(
        raw_score=round_half_up(raw), weighted_score=round_half_up(weighted), confidence=confidence
    )


def _project_component(match_data: Dict) -> ScoreComponent:
    relevance_list = match_data["project_relevance"]
    if not relevance_list:
        return ScoreComponent(raw_score=0.0, weighted_score=0.0, confidence=40.0)
    weights_map = {"High": 100.0, "Medium": 60.0, "Low": 20.0}
    raw = sum(weights_map.get(p.relevance, 0.0) for p in relevance_list) / len(relevance_list)
    weighted = raw * (settings.WEIGHT_PROJECT / 100)
    confidence = 80.0
    return ScoreComponent(
        raw_score=round_half_up(raw), weighted_score=round_half_up(weighted), confidence=confidence
    )


def _experience_component(match_data: Dict) -> ScoreComponent:
    raw = match_data["experience_score"]
    weighted = raw * (settings.WEIGHT_EXPERIENCE / 100)
    return ScoreComponent(
        raw_score=round_half_up(raw), weighted_score=round_half_up(weighted), confidence=70.0
    )


def _education_component(resume_profile, jd_profile) -> ScoreComponent:
    if not jd_profile.education:
        raw = 100.0
        confidence = 50.0
    else:
        required_terms = {e.lower() for e in jd_profile.education}
        candidate_terms = " ".join(resume_profile.education).lower()
        hit = any(term in candidate_terms for term in required_terms)
        raw = 100.0 if hit else 40.0
        confidence = 80.0
    weighted = raw * (settings.WEIGHT_EDUCATION / 100)
    return ScoreComponent(raw_score=round_half_up(raw), weighted_score=round_half_up(weighted), confidence=confidence)


def calculate_score(resume_profile, jd_profile, match_data: Dict) -> ScoreBreakdown:
    """Calculate the full explainable ScoreBreakdown from matching engine output."""
    exact = _exact_component(match_data)
    semantic = _semantic_component(match_data)
    transferable = _transferable_component(match_data)
    project = _project_component(match_data)
    experience = _experience_component(match_data)
    education = _education_component(resume_profile, jd_profile)

    overall = (
        exact.weighted_score
        + semantic.weighted_score
        + transferable.weighted_score
        + project.weighted_score
        + experience.weighted_score
        + education.weighted_score
    )
    overall = clamp(round_half_up(overall))

    return ScoreBreakdown(
        overall_score=overall,
        exact_match=exact,
        semantic=semantic,
        transferable=transferable,
        project=project,
        experience=experience,
        education=education,
    )

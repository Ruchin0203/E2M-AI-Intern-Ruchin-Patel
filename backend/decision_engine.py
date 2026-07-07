"""
decision_engine.py
Deterministic Decision Engine. Contains NO AI/LLM calls.

Acts like a recruiter: converts the score breakdown + matching data into a
decision category, strengths/weaknesses, and a recommendation.
"""

from typing import Dict

from models import ScoreBreakdown, DecisionResult


def _decision_category(overall_score: float) -> str:
    if overall_score >= 90:
        return "Excellent Match"
    if overall_score >= 80:
        return "Strong Match"
    if overall_score >= 70:
        return "Good Match"
    if overall_score >= 55:
        return "Potential Match"
    return "Weak Match"


def _build_strengths(match_data: Dict, score: ScoreBreakdown) -> list:
    strengths = []
    if score.exact_match.raw_score >= 70:
        strengths.append(
            f"Strong direct overlap with {len(match_data['exact_matched'])} required skill(s) matched exactly."
        )
    if score.project.raw_score >= 60:
        high_relevance = [p.project for p in match_data["project_relevance"] if p.relevance == "High"]
        if high_relevance:
            strengths.append(f"Highly relevant project experience: {', '.join(high_relevance[:3])}.")
    if match_data["transferable_skills"]:
        strengths.append(
            f"{len(match_data['transferable_skills'])} transferable skill(s) reduce risk on missing exact matches."
        )
    if score.experience.raw_score >= 80:
        strengths.append("Experience level aligns well with the role's requirements.")
    if not strengths:
        strengths.append("Candidate shows foundational alignment with the role.")
    return strengths


def _build_weaknesses(match_data: Dict, score: ScoreBreakdown) -> list:
    weaknesses = []
    if match_data["missing_skills"]:
        weaknesses.append(
            f"Missing {len(match_data['missing_skills'])} required skill(s): "
            f"{', '.join(match_data['missing_skills'][:5])}."
        )
    if score.project.raw_score < 40:
        weaknesses.append("Limited demonstrated project relevance to this role's responsibilities.")
    if score.experience.raw_score < 50:
        weaknesses.append("Experience level is below what the role typically requires.")
    if not weaknesses:
        weaknesses.append("No major gaps identified.")
    return weaknesses


def _learning_potential(match_data: Dict) -> str:
    transferable_count = len(match_data["transferable_skills"])
    if transferable_count >= 3:
        return "High — candidate has multiple adjacent/transferable skills to build on."
    if transferable_count >= 1:
        return "Moderate — some transferable foundation exists for the missing skills."
    return "Standard — missing skills would need to be learned largely from scratch."


def _recommendation(decision: str) -> str:
    mapping = {
        "Excellent Match": "Fast-track for technical interview.",
        "Strong Match": "Shortlist for technical interview.",
        "Good Match": "Consider for interview; verify missing skills during screening.",
        "Potential Match": "Consider only if role has training runway or skill gaps are acceptable.",
        "Weak Match": "Not recommended for this specific role at this time.",
    }
    return mapping.get(decision, "Review manually.")


def make_decision(match_data: Dict, score: ScoreBreakdown) -> DecisionResult:
    """Apply deterministic recruiter-style decision rules to the score breakdown."""
    decision = _decision_category(score.overall_score)
    strengths = _build_strengths(match_data, score)
    weaknesses = _build_weaknesses(match_data, score)
    learning_potential = _learning_potential(match_data)
    recommendation = _recommendation(decision)

    return DecisionResult(
        decision=decision,
        strengths=strengths,
        weaknesses=weaknesses,
        missing_skills=match_data["missing_skills"],
        transferable_skills=[t.required_skill for t in match_data["transferable_skills"]],
        learning_potential=learning_potential,
        recommendation=recommendation,
    )

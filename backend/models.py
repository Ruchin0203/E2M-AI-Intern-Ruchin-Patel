"""
models.py
Pydantic models used across the application for request/response validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Candidate / Resume Profile
# ---------------------------------------------------------------------------

class ResumeProfile(BaseModel):
    skills: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    libraries: List[str] = Field(default_factory=list)
    cloud: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    experience: List[str] = Field(default_factory=list)
    education: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    domains: List[str] = Field(default_factory=list)
    internship_experience: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    total_experience_years: float = 0.0
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Job Description / Requirement Profile
# ---------------------------------------------------------------------------

class JDProfile(BaseModel):
    must_have_skills: List[str] = Field(default_factory=list)
    nice_to_have_skills: List[str] = Field(default_factory=list)
    responsibilities: List[str] = Field(default_factory=list)
    programming_languages: List[str] = Field(default_factory=list)
    frameworks: List[str] = Field(default_factory=list)
    databases: List[str] = Field(default_factory=list)
    cloud: List[str] = Field(default_factory=list)
    experience_required_years: float = 0.0
    education: List[str] = Field(default_factory=list)
    soft_skills: List[str] = Field(default_factory=list)
    ai_technologies: List[str] = Field(default_factory=list)
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Matching results
# ---------------------------------------------------------------------------

class TransferableSkillMatch(BaseModel):
    resume_skill: str
    required_skill: str
    category: str
    reason: str
    confidence: float


class ProjectRelevance(BaseModel):
    project: str
    relevance: str  # High / Medium / Low
    similarity_percentage: float


class MatchResult(BaseModel):
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    fuzzy_matches: List[dict] = Field(default_factory=list)
    semantic_matches: List[dict] = Field(default_factory=list)
    transferable_skills: List[TransferableSkillMatch] = Field(default_factory=list)
    project_relevance: List[ProjectRelevance] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class ScoreComponent(BaseModel):
    raw_score: float
    weighted_score: float
    confidence: float


class ScoreBreakdown(BaseModel):
    overall_score: float
    exact_match: ScoreComponent
    semantic: ScoreComponent
    transferable: ScoreComponent
    project: ScoreComponent
    experience: ScoreComponent
    education: ScoreComponent


# ---------------------------------------------------------------------------
# Decision Engine
# ---------------------------------------------------------------------------

class DecisionResult(BaseModel):
    decision: str  # Excellent / Strong / Good / Potential / Weak Match
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    transferable_skills: List[str] = Field(default_factory=list)
    learning_potential: str = ""
    recommendation: str = ""


# ---------------------------------------------------------------------------
# Explanation
# ---------------------------------------------------------------------------

class Explanation(BaseModel):
    executive_summary: str = ""
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)
    recruiter_notes: str = ""
    final_recommendation: str = ""


# ---------------------------------------------------------------------------
# API Response
# ---------------------------------------------------------------------------

class AnalyzeResponse(BaseModel):
    overall_score: float
    score_breakdown: ScoreBreakdown
    matched_skills: List[str]
    missing_skills: List[str]
    transferable_skills: List[TransferableSkillMatch]
    project_relevance: List[ProjectRelevance]
    strengths: List[str]
    weaknesses: List[str]
    suggestions: List[str]
    recommendation: str
    explanation: Explanation
    resume_text: str
    jd_text: str
    resume_profile: ResumeProfile
    jd_profile: JDProfile


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None

"""
prompt_templates.py
All LLM prompts live here, kept separate from business logic.
"""

from langchain_core.prompts import PromptTemplate

RESUME_EXTRACTION_TEMPLATE = """You are an expert technical recruiter assistant.
Read the resume text below and extract structured information.

Return ONLY valid JSON with exactly these keys (use empty lists/strings/0 if unknown):

{{
  "skills": [],
  "frameworks": [],
  "libraries": [],
  "cloud": [],
  "databases": [],
  "tools": [],
  "projects": [],
  "experience": [],
  "education": [],
  "certifications": [],
  "soft_skills": [],
  "domains": [],
  "internship_experience": [],
  "achievements": [],
  "total_experience_years": 0
}}

Rules:
- "skills" should include programming languages and general AI/ML skills.
- Split frameworks, libraries, cloud platforms, and databases into their own lists.
- "projects" should be short descriptive titles/summaries of each project (max 20 words each).
- "total_experience_years" is your best numeric estimate of total professional experience in years.
- Do not invent information that is not present in the resume.
- Return raw JSON only. No markdown fences, no commentary, no preamble.

RESUME TEXT:
---
{resume_text}
---
"""

JD_EXTRACTION_TEMPLATE = """You are an expert technical recruiter assistant.
Read the job description below and extract structured hiring requirements.

Return ONLY valid JSON with exactly these keys (use empty lists/strings/0 if unknown):

{{
  "must_have_skills": [],
  "nice_to_have_skills": [],
  "responsibilities": [],
  "programming_languages": [],
  "frameworks": [],
  "databases": [],
  "cloud": [],
  "experience_required_years": 0,
  "education": [],
  "soft_skills": [],
  "ai_technologies": []
}}

Rules:
- Separate mandatory ("must have") requirements from optional ("nice to have") ones based on wording.
- "experience_required_years" should be your best numeric estimate (0 if a fresher role).
- Do not invent requirements that are not implied by the text.
- Return raw JSON only. No markdown fences, no commentary, no preamble.

JOB DESCRIPTION:
---
{jd_text}
---
"""

EXPLANATION_TEMPLATE = """You are a senior technical recruiter writing recruiter notes for a hiring manager.

You are given:
1. The candidate's structured profile
2. The job's structured requirements
3. A deterministic score breakdown that was already calculated using Python (NOT by you)
4. A deterministic decision category that was already calculated using Python (NOT by you)

Your ONLY job is to explain these results in clear, professional, recruiter-style language.
Do NOT recalculate, re-rank, or contradict the score or decision provided.
Do NOT invent skills or facts not present in the provided data.

Return ONLY valid JSON with exactly these keys:

{{
  "executive_summary": "",
  "strengths": [],
  "weaknesses": [],
  "matched_skills": [],
  "missing_skills": [],
  "improvement_suggestions": [],
  "recruiter_notes": "",
  "final_recommendation": ""
}}

CANDIDATE PROFILE (JSON):
{candidate_profile}

JOB REQUIREMENTS (JSON):
{job_requirements}

SCORE BREAKDOWN (JSON, already calculated - do not change):
{score_breakdown}

DECISION (already calculated - do not change):
{decision}

Return raw JSON only. No markdown fences, no commentary, no preamble.
"""


def get_resume_prompt() -> PromptTemplate:
    """Return the PromptTemplate used to extract a structured resume profile."""
    return PromptTemplate(
        template=RESUME_EXTRACTION_TEMPLATE,
        input_variables=["resume_text"],
    )


def get_jd_prompt() -> PromptTemplate:
    """Return the PromptTemplate used to extract structured JD requirements."""
    return PromptTemplate(
        template=JD_EXTRACTION_TEMPLATE,
        input_variables=["jd_text"],
    )


def get_explanation_prompt() -> PromptTemplate:
    """Return the PromptTemplate used to generate the recruiter explanation."""
    return PromptTemplate(
        template=EXPLANATION_TEMPLATE,
        input_variables=["candidate_profile", "job_requirements", "score_breakdown", "decision"],
    )

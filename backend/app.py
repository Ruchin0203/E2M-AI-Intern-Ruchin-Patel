"""
app.py
FastAPI application entrypoint. Wires together:
Parser -> Resume Analyzer -> JD Analyzer -> Matching Engine -> Scoring Engine
-> Decision Engine -> Explanation Chain -> JSON response.
"""

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from config import settings, get_logger
from parser import parse_resume_pdf, PDFParseError
from resume_analyzer import analyze_resume
from jd_analyzer import analyze_job_description
from matcher import run_matching_engine
from scorer import calculate_score
from decision_engine import make_decision
from llm_service import generate_explanation
from utils import clean_text
from models import AnalyzeResponse, Explanation, ErrorResponse

logger = get_logger(__name__)

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Health check / root endpoint."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "ok",
        "llm_configured": settings.USE_LLM,
    }


@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok"}


def _fallback_explanation(match_data: dict, score, decision) -> Explanation:
    """Deterministic explanation used only if the LLM explanation chain is unavailable."""
    summary = (
        f"The candidate is a {decision.decision} with an overall score of "
        f"{score.overall_score}/100. {len(match_data['matched_skills'])} required skill(s) "
        f"were matched and {len(match_data['missing_skills'])} were missing."
    )
    suggestions = [
        f"Consider developing skills in: {', '.join(match_data['missing_skills'][:5])}."
    ] if match_data["missing_skills"] else ["No critical skill gaps identified."]

    return Explanation(
        executive_summary=summary,
        strengths=decision.strengths,
        weaknesses=decision.weaknesses,
        matched_skills=match_data["matched_skills"],
        missing_skills=match_data["missing_skills"],
        improvement_suggestions=suggestions,
        recruiter_notes=f"Learning potential: {decision.learning_potential}",
        final_recommendation=decision.recommendation,
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    resume: UploadFile = File(..., description="Candidate resume PDF"),
    job_description: str = Form(..., description="Job description text"),
):
    """
    Analyze a resume PDF against a pasted job description and return the
    full explainable match report.
    """
    if not job_description or len(job_description.strip()) < 20:
        raise HTTPException(status_code=400, detail="Job description is too short or empty.")

    if resume.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported for the resume.")

    file_bytes = await resume.read()
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(status_code=400, detail=f"Resume exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit.")

    # 1. Parse
    try:
        resume_text = parse_resume_pdf(file_bytes)
    except PDFParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    jd_text = clean_text(job_description)

    # 2. Analyze resume & JD (LangChain / Gemini, with deterministic fallback)
    try:
        resume_profile = analyze_resume(resume_text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Resume analysis failed")
        raise HTTPException(status_code=500, detail=f"Resume analysis failed: {exc}") from exc

    try:
        jd_profile = analyze_job_description(jd_text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("JD analysis failed")
        raise HTTPException(status_code=500, detail=f"Job description analysis failed: {exc}") from exc

    # 3. Matching Engine (deterministic)
    try:
        match_data = run_matching_engine(resume_profile, jd_profile)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Matching engine failed")
        raise HTTPException(status_code=500, detail=f"Matching failed: {exc}") from exc

    # 4. Scoring Engine (deterministic)
    score = calculate_score(resume_profile, jd_profile, match_data)

    # 5. Decision Engine (deterministic)
    decision = make_decision(match_data, score)

    # 6. Explanation Chain (LLM narrative only, with deterministic fallback)
    explanation_raw = generate_explanation(
        candidate_profile=resume_profile.dict(exclude={"raw_text"}),
        job_requirements=jd_profile.dict(exclude={"raw_text"}),
        score_breakdown=score.dict(),
        decision=decision.dict(),
    )

    if explanation_raw:
        try:
            explanation = Explanation(**explanation_raw)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Explanation validation failed (%s); using fallback.", exc)
            explanation = _fallback_explanation(match_data, score, decision)
    else:
        explanation = _fallback_explanation(match_data, score, decision)

    return AnalyzeResponse(
        overall_score=score.overall_score,
        score_breakdown=score,
        matched_skills=match_data["matched_skills"],
        missing_skills=match_data["missing_skills"],
        transferable_skills=match_data["transferable_skills"],
        project_relevance=match_data["project_relevance"],
        strengths=decision.strengths,
        weaknesses=decision.weaknesses,
        suggestions=explanation.improvement_suggestions,
        recommendation=decision.decision,
        explanation=explanation,
        resume_text=resume_text,
        jd_text=jd_text,
        resume_profile=resume_profile,
        jd_profile=jd_profile,
    )

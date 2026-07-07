"""
matcher.py
Deterministic Matching Engine. Contains NO AI/LLM calls.

Implements:
 - Exact skill matching
 - Fuzzy matching (RapidFuzz)
 - Semantic matching (Sentence Transformers cosine similarity)
 - Transferable skill detection
 - Project relevance scoring
 - Experience matching
"""

from typing import List, Dict, Tuple
from functools import lru_cache

from rapidfuzz import fuzz

from config import settings, get_logger
from utils import normalize_skill, dedupe_preserve_order
from transferable_skills import same_category
from models import TransferableSkillMatch, ProjectRelevance

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_embedding_model():
    """Lazily load and cache the sentence-transformers model (loaded once per process)."""
    from sentence_transformers import SentenceTransformer

    logger.info("Loading sentence-transformer model: %s", settings.SENTENCE_TRANSFORMER_MODEL)
    return SentenceTransformer(settings.SENTENCE_TRANSFORMER_MODEL)


def _cosine_similarity(vec_a, vec_b) -> float:
    import numpy as np

    a = np.array(vec_a)
    b = np.array(vec_b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def exact_match(resume_skills: List[str], required_skills: List[str]) -> Tuple[List[str], List[str]]:
    """Return (matched, remaining_unmatched) using case-insensitive exact matching."""
    resume_set = {normalize_skill(s): s for s in resume_skills}
    matched, unmatched = [], []
    for req in required_skills:
        key = normalize_skill(req)
        if key in resume_set:
            matched.append(req)
        else:
            unmatched.append(req)
    return matched, unmatched


def fuzzy_match(
    resume_skills: List[str], required_skills: List[str], threshold: int = None
) -> List[Dict]:
    """
    Return fuzzy matches for required skills against resume skills using
    RapidFuzz token_sort_ratio, above the configured threshold.
    """
    threshold = threshold or settings.FUZZY_MATCH_THRESHOLD
    matches = []
    for req in required_skills:
        best_score, best_skill = 0, None
        for res_skill in resume_skills:
            score = fuzz.token_sort_ratio(normalize_skill(req), normalize_skill(res_skill))
            if score > best_score:
                best_score, best_skill = score, res_skill
        if best_skill and best_score >= threshold:
            matches.append({"required": req, "matched_with": best_skill, "score": best_score})
    return matches


def semantic_match(
    resume_skills: List[str], required_skills: List[str], threshold: float = None
) -> List[Dict]:
    """
    Return semantic matches for required skills against resume skills using
    sentence-transformer embeddings and cosine similarity.
    """
    if not resume_skills or not required_skills:
        return []

    threshold = threshold if threshold is not None else settings.SEMANTIC_MATCH_THRESHOLD
    try:
        model = _get_embedding_model()
        resume_embeddings = model.encode(resume_skills)
        required_embeddings = model.encode(required_skills)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Semantic matching unavailable (%s); skipping.", exc)
        return []

    matches = []
    for i, req in enumerate(required_skills):
        best_score, best_skill = 0.0, None
        for j, res_skill in enumerate(resume_skills):
            score = _cosine_similarity(required_embeddings[i], resume_embeddings[j])
            if score > best_score:
                best_score, best_skill = score, res_skill
        if best_skill and best_score >= threshold:
            matches.append({
                "required": req,
                "matched_with": best_skill,
                "similarity": round(best_score * 100, 1),
            })
    return matches


def transferable_skill_match(
    resume_skills: List[str], required_skills: List[str]
) -> List[TransferableSkillMatch]:
    """Detect category-based transferable skill matches (e.g., Flask <-> FastAPI)."""
    results = []
    matched_required = set()
    for req in required_skills:
        for res_skill in resume_skills:
            if normalize_skill(req) == normalize_skill(res_skill):
                continue  # already an exact match, not transferable
            category = same_category(req, res_skill)
            if category and normalize_skill(req) not in matched_required:
                results.append(
                    TransferableSkillMatch(
                        resume_skill=res_skill,
                        required_skill=req,
                        category=category,
                        reason=f"Both '{res_skill}' and '{req}' belong to the '{category}' category.",
                        confidence=80.0,
                    )
                )
                matched_required.add(normalize_skill(req))
                break
    return results


def project_relevance(
    projects: List[str], responsibilities: List[str], required_skills: List[str]
) -> List[ProjectRelevance]:
    """
    Score each candidate project's relevance to the JD's responsibilities and
    required skills, using semantic similarity. Returns High/Medium/Low labels.
    """
    if not projects:
        return []

    context_sentences = responsibilities + required_skills
    if not context_sentences:
        return [ProjectRelevance(project=p, relevance="Low", similarity_percentage=0.0) for p in projects]

    try:
        model = _get_embedding_model()
        project_embeddings = model.encode(projects)
        context_embeddings = model.encode(context_sentences)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Project relevance semantic scoring unavailable (%s); defaulting to Low.", exc)
        return [ProjectRelevance(project=p, relevance="Low", similarity_percentage=0.0) for p in projects]

    results = []
    for i, project in enumerate(projects):
        best_score = 0.0
        for j in range(len(context_sentences)):
            score = _cosine_similarity(project_embeddings[i], context_embeddings[j])
            best_score = max(best_score, score)
        pct = round(best_score * 100, 1)
        if pct >= 65:
            label = "High"
        elif pct >= 40:
            label = "Medium"
        else:
            label = "Low"
        results.append(ProjectRelevance(project=project, relevance=label, similarity_percentage=pct))
    return results


def experience_match_score(
    candidate_years: float, required_years: float, project_relevance_scores: List[ProjectRelevance]
) -> float:
    """
    Compute a 0-100 experience match score. Freshers with highly relevant
    projects are not penalized heavily even with limited formal experience.
    """
    if required_years <= 0:
        base = 100.0
    else:
        ratio = candidate_years / required_years if required_years > 0 else 1.0
        base = min(100.0, ratio * 100.0)

    if project_relevance_scores:
        high_relevance_count = sum(1 for p in project_relevance_scores if p.relevance == "High")
        project_boost = min(30.0, high_relevance_count * 10.0)
        base = min(100.0, base + project_boost if candidate_years < required_years else base)

    return round(base, 1)


def run_matching_engine(resume_profile, jd_profile) -> Dict:
    """
    Orchestrate the full deterministic matching pipeline and return all
    intermediate matching artifacts needed by the scoring engine.
    """
    resume_all_skills = dedupe_preserve_order(
        resume_profile.skills
        + resume_profile.frameworks
        + resume_profile.libraries
        + resume_profile.cloud
        + resume_profile.databases
        + resume_profile.tools
    )
    required_skills = dedupe_preserve_order(
        jd_profile.must_have_skills
        + jd_profile.programming_languages
        + jd_profile.frameworks
        + jd_profile.cloud
        + jd_profile.databases
        + jd_profile.ai_technologies
    )
    nice_to_have = dedupe_preserve_order(jd_profile.nice_to_have_skills)

    exact_matched, remaining_after_exact = exact_match(resume_all_skills, required_skills)

    fuzzy_matches = fuzzy_match(resume_all_skills, remaining_after_exact)
    fuzzy_matched_names = [m["required"] for m in fuzzy_matches]
    remaining_after_fuzzy = [r for r in remaining_after_exact if r not in fuzzy_matched_names]

    semantic_matches = semantic_match(resume_all_skills, remaining_after_fuzzy)
    semantic_matched_names = [m["required"] for m in semantic_matches]
    remaining_after_semantic = [r for r in remaining_after_fuzzy if r not in semantic_matched_names]

    transferable = transferable_skill_match(resume_all_skills, remaining_after_semantic)
    transferable_matched_names = [t.required_skill for t in transferable]

    missing_skills = [
        r for r in remaining_after_semantic
        if normalize_skill(r) not in {normalize_skill(t) for t in transferable_matched_names}
    ]

    all_matched_skills = dedupe_preserve_order(
        exact_matched + fuzzy_matched_names + semantic_matched_names
    )

    proj_relevance = project_relevance(
        resume_profile.projects, jd_profile.responsibilities, required_skills
    )

    experience_score = experience_match_score(
        resume_profile.total_experience_years,
        jd_profile.experience_required_years,
        proj_relevance,
    )

    return {
        "resume_all_skills": resume_all_skills,
        "required_skills": required_skills,
        "nice_to_have_skills": nice_to_have,
        "exact_matched": exact_matched,
        "fuzzy_matches": fuzzy_matches,
        "semantic_matches": semantic_matches,
        "transferable_skills": transferable,
        "matched_skills": all_matched_skills,
        "missing_skills": missing_skills,
        "project_relevance": proj_relevance,
        "experience_score": experience_score,
    }

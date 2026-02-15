from __future__ import annotations

from app.schemas.gap_analysis import InputValidationResponse
from app.services.jd_clipping import clip_job_description
from app.services.skill_taxonomy import extract_skills, normalize_skill_text
from app.services.text_processing import normalize_text, word_count

MIN_WORDS = 50
MAX_VALIDATION_ERROR_MESSAGE_CHARS = 300


def validate_input_quality_raw(resume_text: str, jd_text: str) -> InputValidationResponse:
    resume_words = word_count(resume_text)
    jd_words = word_count(jd_text)
    resume_skills = {normalize_skill_text(s) for s in extract_skills(resume_text) if normalize_skill_text(s)}
    jd_skills = {normalize_skill_text(s) for s in extract_skills(jd_text) if normalize_skill_text(s)}
    resume_tech = len(resume_skills)
    jd_tech = len(jd_skills)

    resume_valid = resume_words >= MIN_WORDS
    jd_valid = jd_words >= MIN_WORDS

    error_message: str | None = None
    if not resume_valid or not jd_valid:
        reasons: list[str] = []
        if not resume_valid:
            reasons.append(f"Resume length insufficient ({resume_words} words)")
        if not jd_valid:
            reasons.append(f"JD length insufficient ({jd_words} words)")
        error_message = "; ".join(reasons)[:MAX_VALIDATION_ERROR_MESSAGE_CHARS]

    return InputValidationResponse(
        is_valid=resume_valid and jd_valid,
        error_message=error_message,
        resume_word_count=resume_words,
        jd_word_count=jd_words,
        resume_tech_entities=resume_tech,
        jd_tech_entities=jd_tech,
    )


def prepare_texts_for_skill_extraction(resume_text: str, jd_text: str) -> tuple[str, str, str]:
    clipped_jd, clean_strategy = clip_job_description(jd_text)
    normalized_resume = normalize_text(resume_text)
    normalized_jd = normalize_text(clipped_jd)
    return normalized_resume, normalized_jd, clean_strategy


def validate_input_quality(resume_text: str, jd_text: str) -> InputValidationResponse:
    normalized_resume, normalized_jd, _ = prepare_texts_for_skill_extraction(resume_text, jd_text)
    return validate_input_quality_raw(normalized_resume, normalized_jd)

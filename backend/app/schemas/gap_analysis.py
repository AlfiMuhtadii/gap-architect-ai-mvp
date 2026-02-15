from pydantic import BaseModel, Field, field_validator
from typing import Any
from uuid import UUID
from app.models.gap import GapAnalysisStatus
from app.core.config import settings


class GapAnalysisCreate(BaseModel):
    resume_text: str = Field(min_length=1, max_length=settings.max_resume_chars)
    jd_text: str = Field(min_length=1, max_length=settings.max_jd_chars)
    model: str | None = None
    prompt_version: str | None = None

    @field_validator("resume_text", "jd_text", mode="before")
    @classmethod
    def _strip_and_require_text(cls, v: str) -> str:
        if not isinstance(v, str):
            raise ValueError("must be a string")
        cleaned = " ".join(v.split())
        if not cleaned:
            raise ValueError("must be non-empty")
        return cleaned

    @field_validator("model", "prompt_version", mode="before")
    @classmethod
    def _validate_identifiers(cls, v: str | None) -> str | None:
        if v is None:
            return None
        if not isinstance(v, str):
            raise ValueError("must be a string")
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("must be non-empty")
        import re
        if not re.fullmatch(r"[A-Za-z0-9._:-]{1,64}", cleaned):
            raise ValueError("invalid format")
        return cleaned

class ActionStepOut(BaseModel):
    title: str
    why: str
    deliverable: str


class InterviewQuestionOut(BaseModel):
    question: str
    focus_gap: str
    what_good_looks_like: str


class GapResultOut(BaseModel):
    missing_skills: list[str]
    action_steps: list[ActionStepOut]
    interview_questions: list[InterviewQuestionOut]
    roadmap_markdown: str
    match_percent: float | None = None
    match_reason: str | None = None
    metadata: dict[str, Any] | None = None
    generation_meta: dict[str, Any] | None = None

    class Config:
        from_attributes = True


class GapAnalysisOut(BaseModel):
    id: UUID
    status: GapAnalysisStatus
    result: GapResultOut | None = None
    error_message: str | None = None

    class Config:
        from_attributes = True


class InputValidationRequest(BaseModel):
    resume_text: str = Field(min_length=1, max_length=settings.max_resume_chars)
    jd_text: str = Field(min_length=1, max_length=settings.max_jd_chars)


class InputValidationResponse(BaseModel):
    is_valid: bool
    error_message: str | None = None
    resume_word_count: int
    jd_word_count: int
    resume_tech_entities: int
    jd_tech_entities: int

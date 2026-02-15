import hashlib
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from uuid import UUID

from fastapi import BackgroundTasks
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.gap import GapAnalysis, GapAnalysisStatus, GapResult, JdCleanRun, JdCleanStatus, LlmRun
from app.schemas.gap_analysis import GapAnalysisCreate, GapAnalysisOut, GapResultOut, InputValidationResponse
from app.core.config import settings
from app.services import llm_service
from app.services.input_validation import (
    prepare_texts_for_skill_extraction,
    validate_input_quality as validate_input_quality_processed,
    validate_input_quality_raw,
)
from app.services.skill_taxonomy import get_skill_taxonomy_map, normalize_skill_text
from app.services.text_processing import normalize_text

_LAST_SWEEP_AT: datetime | None = None
_SWEEP_INTERVAL_SECONDS = 15
logger = logging.getLogger("app.gap_analysis_service")


# Backward-compatible helpers (referenced by existing tests/imports)
def _normalize_text(text: str) -> str:
    return normalize_text(text)


def _validate_input_quality(resume_text: str, jd_text: str) -> InputValidationResponse:
    return validate_input_quality_raw(resume_text, jd_text)


def _prepare_texts_for_skill_extraction(resume_text: str, jd_text: str) -> tuple[str, str, str]:
    return prepare_texts_for_skill_extraction(resume_text, jd_text)


def validate_input_quality(resume_text: str, jd_text: str) -> InputValidationResponse:
    return validate_input_quality_processed(resume_text, jd_text)


def _fingerprint(resume_text: str, jd_text: str) -> str:
    payload = f"{resume_text}\n---\n{jd_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


async def _sweep_stuck_pending(session: AsyncSession) -> None:
    global _LAST_SWEEP_AT

    ttl = int(getattr(settings, "pending_timeout_seconds", 0) or 0)
    if ttl <= 0:
        return
    now = datetime.now(timezone.utc)
    if _LAST_SWEEP_AT is not None:
        elapsed = (now - _LAST_SWEEP_AT).total_seconds()
        if elapsed < _SWEEP_INTERVAL_SECONDS:
            return
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=ttl)
    try:
        await session.execute(
            update(GapAnalysis)
            .where(
                GapAnalysis.status == GapAnalysisStatus.PENDING,
                GapAnalysis.updated_at < cutoff,
            )
            .values(
                status=GapAnalysisStatus.FAILED_TIMEOUT,
                error_message="processing_timeout",
            )
        )
        _LAST_SWEEP_AT = now
    except Exception as exc:  # noqa: BLE001
        # Fail-open: timeout sweep should never block request flow.
        logger.warning(
            "sweep_error_suppressed",
            extra={"error": str(exc)},
            exc_info=True,
        )
        return


async def _upsert_jd_clean_run(
    session: AsyncSession,
    *,
    input_hash: str,
    clean_strategy: str,
    status: JdCleanStatus,
    error_message: str | None,
) -> None:
    stmt = insert(JdCleanRun).values(
        input_hash=input_hash,
        clean_strategy=clean_strategy,
        status=status,
        error_message=error_message,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[JdCleanRun.input_hash],
        set_={
            "clean_strategy": clean_strategy,
            "status": status,
            "error_message": error_message,
        },
    )
    await session.execute(stmt)


def _build_result_out(result: GapResult | None) -> GapResultOut | None:
    if not isinstance(result, GapResult):
        return None
    taxonomy_map = get_skill_taxonomy_map()
    missing = result.missing_skills if isinstance(result.missing_skills, list) else []
    unmapped = sorted(
        {
            str(skill).strip()
            for skill in missing
            if str(skill).strip() and normalize_skill_text(str(skill).strip()) not in taxonomy_map
        }
    )
    return GapResultOut(
        missing_skills=result.missing_skills,
        action_steps=result.action_steps,
        interview_questions=result.interview_questions,
        roadmap_markdown=result.roadmap_markdown,
        match_percent=result.match_percent,
        match_reason=result.match_reason,
        metadata={"unmapped_count": len(unmapped), "unmapped_list": unmapped},
    )


def _build_generation_meta(analysis: GapAnalysis) -> dict | None:
    runs = list(getattr(analysis, "llm_runs", []) or [])
    if not runs:
        return None
    latest: LlmRun = max(runs, key=lambda r: r.created_at or 0)
    response_json = latest.response_json if isinstance(latest.response_json, dict) else {}
    meta = response_json.get("_meta") if isinstance(response_json, dict) else None
    out = {
        "provider": latest.provider,
        "model": latest.model,
        "status": str(latest.status.value if hasattr(latest.status, "value") else latest.status),
        "duration_ms": latest.duration_ms,
        "created_at": latest.created_at.isoformat() if latest.created_at else None,
    }
    if isinstance(meta, dict):
        out.update(meta)
    return out


async def create_or_get_gap_analysis(
    session: AsyncSession,
    payload: GapAnalysisCreate,
    background_tasks: Optional[BackgroundTasks] = None,
) -> GapAnalysisOut:
    await _sweep_stuck_pending(session)
    raw_resume_norm = _normalize_text(payload.resume_text)
    raw_jd_norm = _normalize_text(payload.jd_text)
    raw_fingerprint = _fingerprint(raw_resume_norm, raw_jd_norm)

    normalized_resume, normalized_jd, clean_strategy = _prepare_texts_for_skill_extraction(
        payload.resume_text, payload.jd_text
    )
    validation = _validate_input_quality(normalized_resume, normalized_jd)
    if not validation.is_valid:
        error_message = validation.error_message or "Input validation failed"
        await _upsert_jd_clean_run(
            session,
            input_hash=raw_fingerprint,
            clean_strategy="validation_failed_short_input",
            status=JdCleanStatus.FAILED,
            error_message=error_message,
        )
        await session.commit()
        raise ValueError(error_message)

    if not normalized_resume or not normalized_jd:
        raise ValueError("Resume and JD must contain alphanumeric characters")
    fingerprint = _fingerprint(normalized_resume, normalized_jd)

    await _upsert_jd_clean_run(
        session,
        input_hash=fingerprint,
        clean_strategy=clean_strategy,
        status=JdCleanStatus.SUCCESS,
        error_message=None,
    )

    stmt = (
        select(GapAnalysis)
        .where(GapAnalysis.fingerprint == fingerprint)
        .options(selectinload(GapAnalysis.gap_result), selectinload(GapAnalysis.llm_runs))
    )
    existing = (await session.execute(stmt)).scalars().first()

    if existing and existing.status == GapAnalysisStatus.DONE:
        result_out = _build_result_out(existing.gap_result)
        if result_out is not None:
            result_out.generation_meta = _build_generation_meta(existing)
        return GapAnalysisOut(
            id=existing.id,
            status=existing.status,
            result=result_out,
            error_message=existing.error_message,
        )

    if existing and existing.status == GapAnalysisStatus.PENDING:
        return GapAnalysisOut(id=existing.id, status=existing.status, result=None, error_message=existing.error_message)

    if existing and existing.status in (
        GapAnalysisStatus.FAILED_LLM,
        GapAnalysisStatus.FAILED_VALIDATION,
        GapAnalysisStatus.FAILED_TIMEOUT,
    ):
        cooldown = int(getattr(settings, "retry_cooldown_seconds", 0) or 0)
        if cooldown > 0 and existing.updated_at is not None:
            updated_at = existing.updated_at
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - updated_at).total_seconds()
            if elapsed < cooldown:
                wait_seconds = max(int(cooldown - elapsed), 1)
                return GapAnalysisOut(
                    id=existing.id,
                    status=existing.status,
                    result=None,
                    error_message=f"Retry cooldown active. Please retry in {wait_seconds}s.",
                )

        transition = await session.execute(
            update(GapAnalysis)
            .where(
                GapAnalysis.id == existing.id,
                GapAnalysis.status.in_(
                    [
                        GapAnalysisStatus.FAILED_LLM,
                        GapAnalysisStatus.FAILED_VALIDATION,
                        GapAnalysisStatus.FAILED_TIMEOUT,
                    ]
                ),
            )
            .values(status=GapAnalysisStatus.PENDING, error_message=None)
        )
        await session.commit()
        if (transition.rowcount or 0) == 0:
            current = (
                await session.execute(select(GapAnalysis).where(GapAnalysis.id == existing.id))
            ).scalars().first()
            if current is None:
                raise ValueError("gap_analysis not found during retry transition")
            return GapAnalysisOut(
                id=current.id,
                status=current.status,
                result=None,
                error_message=current.error_message,
            )
        if background_tasks is not None:
            background_tasks.add_task(llm_service.process_gap_analysis, existing.id)
        else:
            await llm_service.process_gap_analysis(existing.id)
        return GapAnalysisOut(id=existing.id, status=GapAnalysisStatus.PENDING, result=None, error_message=None)

    gap_analysis = GapAnalysis(
        fingerprint=fingerprint,
        resume_text=normalized_resume,
        jd_text=normalized_jd,
        status=GapAnalysisStatus.PENDING,
        model=(payload.model or settings.llm_model or "requested-default"),
        prompt_version=f"{(payload.prompt_version or 'v1')}:{clean_strategy}",
    )
    session.add(gap_analysis)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        existing = (await session.execute(stmt)).scalars().first()
        if existing and existing.status == GapAnalysisStatus.DONE:
            result_out = _build_result_out(existing.gap_result)
            if result_out is not None:
                result_out.generation_meta = _build_generation_meta(existing)
            return GapAnalysisOut(
                id=existing.id,
                status=existing.status,
                result=result_out,
                error_message=existing.error_message,
            )
        if existing:
            return GapAnalysisOut(id=existing.id, status=existing.status, result=None, error_message=existing.error_message)
        raise
    await session.refresh(gap_analysis)

    if background_tasks is not None:
        background_tasks.add_task(llm_service.process_gap_analysis, gap_analysis.id)
    else:
        await llm_service.process_gap_analysis(gap_analysis.id)

    return GapAnalysisOut(id=gap_analysis.id, status=gap_analysis.status, result=None, error_message=gap_analysis.error_message)


async def get_gap_analysis(session: AsyncSession, gap_analysis_id: UUID) -> GapAnalysisOut | None:
    # Keep polling clients informed: stale PENDING rows are transitioned to FAILED_TIMEOUT.
    await _sweep_stuck_pending(session)
    await session.commit()

    stmt = (
        select(GapAnalysis)
        .where(GapAnalysis.id == gap_analysis_id)
        .options(selectinload(GapAnalysis.gap_result), selectinload(GapAnalysis.llm_runs))
    )
    analysis = (await session.execute(stmt)).scalars().first()
    if analysis is None:
        return None
    result = analysis.gap_result
    result_out = (
        GapResultOut(
            missing_skills=result.missing_skills,
            action_steps=result.action_steps,
            interview_questions=result.interview_questions,
            roadmap_markdown=result.roadmap_markdown,
            match_percent=result.match_percent,
            match_reason=result.match_reason,
            generation_meta=_build_generation_meta(analysis),
        )
        if isinstance(result, GapResult)
        else None
    )
    return GapAnalysisOut(id=analysis.id, status=analysis.status, result=result_out, error_message=analysis.error_message)

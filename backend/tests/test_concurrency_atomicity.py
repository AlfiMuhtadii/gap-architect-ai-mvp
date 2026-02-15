import asyncio

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core import config as config_module
from app.models.gap import GapAnalysis, GapAnalysisStatus, GapResult, JdCleanRun, LlmRun
from app.schemas.gap_analysis import GapAnalysisCreate
from app.services import llm_service
from app.services import gap_analysis_ai
from app.services.gap_analysis_service import (
    _fingerprint,
    _prepare_texts_for_skill_extraction,
    create_or_get_gap_analysis,
)


async def _reset_tables(maker: async_sessionmaker[AsyncSession]) -> None:
    async with maker() as session:
        for table in reversed(GapAnalysis.metadata.sorted_tables):
            await session.execute(table.delete())
        await session.commit()


@pytest.mark.asyncio
async def test_retry_transition_atomic_single_winner(engine, monkeypatch):
    monkeypatch.setattr(config_module.settings, "retry_cooldown_seconds", 0)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await _reset_tables(maker)
    calls = {"count": 0}

    async def _fake_process(_gap_analysis_id):
        calls["count"] += 1
        await asyncio.sleep(0.01)

    monkeypatch.setattr(llm_service, "process_gap_analysis", _fake_process)

    resume_text = " ".join(["python"] * 55)
    jd_text = " ".join(["kubernetes"] * 55)
    normalized_resume, normalized_jd, _ = _prepare_texts_for_skill_extraction(resume_text, jd_text)
    fp = _fingerprint(normalized_resume, normalized_jd)

    async with maker() as session:
        session.add(
            GapAnalysis(
                fingerprint=fp,
                resume_text=normalized_resume,
                jd_text=normalized_jd,
                status=GapAnalysisStatus.FAILED_LLM,
                model="m",
                prompt_version="v1",
            )
        )
        await session.commit()

    payload = GapAnalysisCreate(
        resume_text=resume_text,
        jd_text=jd_text,
        model="m",
        prompt_version="v1",
    )

    async def _run_once():
        async with maker() as session:
            return await create_or_get_gap_analysis(session, payload, background_tasks=None)

    out1, out2 = await asyncio.gather(_run_once(), _run_once())
    assert out1.status == GapAnalysisStatus.PENDING
    assert out2.status in (GapAnalysisStatus.PENDING, GapAnalysisStatus.FAILED_LLM, GapAnalysisStatus.DONE)
    assert calls["count"] == 1


@pytest.mark.asyncio
async def test_jd_clean_run_upsert_atomic_on_parallel_validation_fail(engine):
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await _reset_tables(maker)
    payload = GapAnalysisCreate(
        resume_text="short input",
        jd_text="short jd",
        model="m",
        prompt_version="v1",
    )

    async def _run_once():
        async with maker() as session:
            with pytest.raises(ValueError):
                await create_or_get_gap_analysis(session, payload, background_tasks=None)

    await asyncio.gather(_run_once(), _run_once(), _run_once())

    async with maker() as session:
        rows = (await session.execute(select(JdCleanRun))).scalars().all()
        assert len(rows) == 1
        assert rows[0].status.value == "FAILED"


class _ValidProvider:
    name = "valid"
    model = "valid-1"
    last_usage = None

    async def generate(self, prompt: str) -> str:
        return (
            '{'
            '"missing_skills":["a","b"],'
            '"top_priority_skills":["a"],'
            '"action_steps":[{"title":"t1","why":"w1","deliverable":"d1"},'
            '{"title":"t2","why":"w2","deliverable":"d2"},'
            '{"title":"t3","why":"w3","deliverable":"d3"}],'
            '"interview_questions":[{"question":"q1","focus_gap":"g1","what_good_looks_like":"w1"},'
            '{"question":"q2","focus_gap":"g2","what_good_looks_like":"w2"},'
            '{"question":"q3","focus_gap":"g3","what_good_looks_like":"w3"}],'
            '"roadmap_markdown":"## Gap Summary\\nGood baseline.\\n\\n## Priority Skills to Learn\\n- a\\n- b\\n- c\\n\\n## Concrete Steps\\n### Step 1: t1\\n**Why:** w1\\n**Deliverable:** d1\\n### Step 2: t2\\n**Why:** w2\\n**Deliverable:** d2\\n### Step 3: t3\\n**Why:** w3\\n**Deliverable:** d3\\n\\n## Expected Outcomes / Readiness\\n- o1\\n- o2\\n\\n## Suggested Learning Order\\n1. a\\n2. b\\n3. c",'
            '"match_percent":75.5,'
            '"match_reason":"Matched 3 of 4 skills"'
            '}'
        )


@pytest.mark.asyncio
async def test_concurrent_success_cas_single_gap_result_and_llm_run(engine):
    if engine.dialect.name == "sqlite":
        pytest.xfail("SQLite in-memory concurrency is nondeterministic for this CAS path")
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await _reset_tables(maker)

    async with maker() as session:
        analysis = GapAnalysis(
            fingerprint="x" * 64,
            resume_text="mongodb mysql redis sqlite rag",
            jd_text="c# dotnet core rust kubernetes",
            status=GapAnalysisStatus.PENDING,
            model="m",
            prompt_version="v1",
        )
        session.add(analysis)
        await session.commit()
        analysis_id = analysis.id

    async def _run_once():
        async with maker() as session:
            await gap_analysis_ai.run_gap_analysis_ai(
                session,
                analysis_id,
                _ValidProvider(),
                prompt="x",
            )

    await asyncio.gather(_run_once(), _run_once())

    async with maker() as session:
        results = (await session.execute(select(GapResult).where(GapResult.gap_analysis_id == analysis_id))).scalars().all()
        runs = (await session.execute(select(LlmRun).where(LlmRun.gap_analysis_id == analysis_id))).scalars().all()
        row = (await session.execute(select(GapAnalysis).where(GapAnalysis.id == analysis_id))).scalars().first()
        assert row is not None, "missing analysis row"
        assert len(results) == 1, f"status={row.status} err={row.error_message} runs={len(runs)}"
        assert len(runs) == 1, f"status={row.status} err={row.error_message}"
        assert row.status == GapAnalysisStatus.DONE


@pytest.mark.asyncio
async def test_crash_between_persist_and_log_rolls_back_success(engine, monkeypatch):
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await _reset_tables(maker)

    async with maker() as session:
        analysis = GapAnalysis(
            fingerprint="y" * 64,
            resume_text="mongodb mysql redis sqlite rag",
            jd_text="c# dotnet core rust kubernetes",
            status=GapAnalysisStatus.PENDING,
            model="m",
            prompt_version="v1",
        )
        session.add(analysis)
        await session.commit()
        analysis_id = analysis.id

    async def _boom(*args, **kwargs):
        raise RuntimeError("forced_log_failure")

    monkeypatch.setattr(gap_analysis_ai, "_log_llm_run", _boom)

    async with maker() as session:
        await gap_analysis_ai.run_gap_analysis_ai(
            session,
            analysis_id,
            _ValidProvider(),
            prompt="x",
        )

    async with maker() as session:
        results = (await session.execute(select(GapResult).where(GapResult.gap_analysis_id == analysis_id))).scalars().all()
        row = (await session.execute(select(GapAnalysis).where(GapAnalysis.id == analysis_id))).scalars().first()
        assert len(results) == 0
        assert row is not None and row.status == GapAnalysisStatus.FAILED_LLM

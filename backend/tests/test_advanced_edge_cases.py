import pytest
from sqlalchemy import select, delete
from app.core import config as config_module
from app.models.gap import GapAnalysis, GapAnalysisStatus, GapResult
from app.services import llm_service
from app.services.gap_analysis_ai import run_gap_analysis_ai
from app.services.gap_analysis_service import _fingerprint, _normalize_text
from app.services.skill_matcher import match_skills


@pytest.mark.asyncio
async def test_retry_failed_status_resets_and_reprocesses(client, db_session, monkeypatch):
    monkeypatch.setattr(config_module.settings, "retry_cooldown_seconds", 0)
    called = {"count": 0}

    async def _fake_process(_gap_analysis_id):
        called["count"] += 1

    monkeypatch.setattr(llm_service, "process_gap_analysis", _fake_process)

    payload = {
        "resume_text": " ".join(["python"] * 55),
        "jd_text": " ".join(["docker"] * 55),
        "model": "m",
        "prompt_version": "v1",
    }
    fp = _fingerprint(_normalize_text(payload["resume_text"]), _normalize_text(payload["jd_text"]))
    await db_session.execute(delete(GapAnalysis).where(GapAnalysis.fingerprint == fp))
    await db_session.commit()
    analysis = GapAnalysis(
        fingerprint=fp,
        resume_text=_normalize_text(payload["resume_text"]),
        jd_text=_normalize_text(payload["jd_text"]),
        status=GapAnalysisStatus.FAILED_LLM,
        model="m",
        prompt_version="v1",
    )
    db_session.add(analysis)
    await db_session.commit()

    res = await client.post("/api/v1/gap-analyses", json=payload)
    assert res.status_code == 201
    body = res.json()
    assert body["status"] == "PENDING"
    assert called["count"] == 1


@pytest.mark.asyncio
async def test_retry_failed_status_respects_cooldown(client, db_session, monkeypatch):
    monkeypatch.setattr(config_module.settings, "retry_cooldown_seconds", 15)
    called = {"count": 0}

    async def _fake_process(_gap_analysis_id):
        called["count"] += 1

    monkeypatch.setattr(llm_service, "process_gap_analysis", _fake_process)

    payload = {
        "resume_text": " ".join(["python"] * 55),
        "jd_text": " ".join(["docker"] * 55),
        "model": "m",
        "prompt_version": "v1",
    }
    fp = _fingerprint(_normalize_text(payload["resume_text"]), _normalize_text(payload["jd_text"]))
    await db_session.execute(delete(GapAnalysis).where(GapAnalysis.fingerprint == fp))
    await db_session.commit()
    analysis = GapAnalysis(
        fingerprint=fp,
        resume_text=_normalize_text(payload["resume_text"]),
        jd_text=_normalize_text(payload["jd_text"]),
        status=GapAnalysisStatus.FAILED_LLM,
        model="m",
        prompt_version="v1",
    )
    db_session.add(analysis)
    await db_session.commit()

    res = await client.post("/api/v1/gap-analyses", json=payload)
    assert res.status_code == 201
    body = res.json()
    assert body["status"] == "FAILED_LLM"
    assert "Retry cooldown active" in (body.get("error_message") or "")
    assert called["count"] == 0


@pytest.mark.asyncio
async def test_match_skills_token_fallback_without_taxonomy(monkeypatch):
    monkeypatch.setattr("app.services.skill_matcher.extract_skills", lambda _text: [])
    missing, match_percent, reason, *_ = match_skills(
        resume_text="python docker",
        jd_text="python docker kubernetes",
    )
    assert "Fallback token match" in reason
    assert isinstance(missing, list)
    assert match_percent >= 0


def test_prompt_truncation(monkeypatch):
    from app.services import llm_service as llm_module

    original = config_module.settings.max_prompt_chars
    monkeypatch.setattr(config_module.settings, "max_prompt_chars", 100)
    prompt = llm_module._build_prompt("a" * 200, "b" * 200)
    assert len(prompt) <= 100
    monkeypatch.setattr(config_module.settings, "max_prompt_chars", original)


@pytest.mark.asyncio
async def test_no_jd_requirements_short_circuits_guidance(db_session):
    class _FakeProvider:
        name = "openai_compatible"
        model = "fake-model"
        last_usage = {}

        async def generate(self, prompt: str, *, temperature: float | None = None) -> str:
            _ = prompt, temperature
            return (
                "{"
                "\"jd_skills_extracted\": [\"python\", \"docker\", \"kubernetes\"],"
                "\"resume_skills_extracted\": [\"python\", \"docker\"],"
                "\"missing_skills\": [\"kubernetes\"],"
                "\"top_priority_skills\": [\"kubernetes\"],"
                "\"action_steps\": ["
                "{\"title\":\"Build Kubernetes demo\",\"why\":\"practice\",\"deliverable\":\"repo\"},"
                "{\"title\":\"Add observability\",\"why\":\"practice\",\"deliverable\":\"dashboard\"},"
                "{\"title\":\"Run load test\",\"why\":\"practice\",\"deliverable\":\"report\"}"
                "],"
                "\"interview_questions\": ["
                "{\"question\":\"Q1\",\"focus_gap\":\"kubernetes\",\"what_good_looks_like\":\"A\"},"
                "{\"question\":\"Q2\",\"focus_gap\":\"kubernetes\",\"what_good_looks_like\":\"A\"},"
                "{\"question\":\"Q3\",\"focus_gap\":\"kubernetes\",\"what_good_looks_like\":\"A\"}"
                "],"
                "\"roadmap_markdown\":\"## Gap Summary\\nSummary.\\n\\n## Priority Skills to Learn\\n- kubernetes\\n\\n## Concrete Steps\\n### Step 1 Do A\\n**Why:** one.\\n**Deliverable:** one.\\n\\n### Step 2 Do B\\n**Why:** two.\\n**Deliverable:** two.\\n\\n### Step 3 Do C\\n**Why:** three.\\n**Deliverable:** three.\\n\\n## Expected Outcomes / Readiness\\n- outcome\\n\\n## Suggested Learning Order\\n1. kubernetes\","
                "\"match_percent\": 10.0,"
                "\"match_reason\": \"hallucinated\""
                "}"
            )

    analysis = GapAnalysis(
        fingerprint="f" * 64,
        resume_text="python docker aws kubernetes observability testing ci cd backend",
        jd_text="aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj",
        status=GapAnalysisStatus.PENDING,
        model="m",
        prompt_version="v1",
    )
    db_session.add(analysis)
    await db_session.commit()
    await db_session.refresh(analysis)

    await run_gap_analysis_ai(
        db_session,
        analysis.id,
        _FakeProvider(),
        "prompt",
    )

    fresh = (
        await db_session.execute(select(GapAnalysis).where(GapAnalysis.id == analysis.id))
    ).scalars().first()
    assert fresh is not None
    assert fresh.status == GapAnalysisStatus.DONE

    gap = (
        await db_session.execute(select(GapResult).where(GapResult.gap_analysis_id == analysis.id))
    ).scalars().first()
    assert gap is not None
    assert gap.missing_skills == []
    assert gap.top_priority_skills == []
    assert gap.action_steps == []
    assert gap.interview_questions == []
    assert gap.match_percent == 100.0
    assert "No identifiable technical requirements" in (gap.match_reason or "")

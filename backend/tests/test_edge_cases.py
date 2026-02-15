import asyncio
import pytest

from app.core import config as config_module
from app.models.gap import GapAnalysis, GapAnalysisStatus
from app.services.gap_analysis_ai import run_gap_analysis_ai
from .factories import GapAnalysisPayloadFactory, make_gap_analysis


@pytest.mark.asyncio
async def test_empty_input_422(client):
    payload = {"resume_text": "", "jd_text": "", "model": "m", "prompt_version": "v1"}
    res = await client.post("/api/v1/gap-analyses", json=payload)
    assert res.status_code == 422


@pytest.mark.asyncio
async def test_large_input_rejected(client):
    payload = GapAnalysisPayloadFactory(
        resume_text="a" * (config_module.settings.max_resume_chars + 1),
        jd_text="b" * (config_module.settings.max_jd_chars + 1),
    ).build()
    res = await client.post("/api/v1/gap-analyses", json=payload)
    assert res.status_code == 422


class SlowProvider:
    name = "slow"
    model = "slow-1"

    async def generate(self, prompt: str) -> str:
        await asyncio.sleep(0.2)
        return "{}"


@pytest.mark.asyncio
async def test_llm_timeout_failed_llm(db_session, monkeypatch):
    monkeypatch.setattr(config_module.settings, "llm_timeout_seconds", 0.05)
    analysis = make_gap_analysis()
    db_session.add(analysis)
    await db_session.commit()

    await run_gap_analysis_ai(db_session, analysis.id, SlowProvider(), prompt="x")

    assert analysis.status == GapAnalysisStatus.FAILED_LLM

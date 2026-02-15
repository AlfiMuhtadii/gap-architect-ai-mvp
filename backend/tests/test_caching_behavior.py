import pytest
from uuid import UUID
from sqlalchemy import select

from app.models.gap import GapAnalysis, GapAnalysisStatus
from app.services import llm_service
from .factories import GapAnalysisPayloadFactory, make_gap_result


@pytest.mark.asyncio
async def test_caching_behavior(client, db_session, monkeypatch):
    called = {"count": 0}

    async def _fake_process(_gap_analysis_id):
        called["count"] += 1

    monkeypatch.setattr(llm_service, "process_gap_analysis", _fake_process)

    payload = GapAnalysisPayloadFactory().build()
    res1 = await client.post("/api/v1/gap-analyses", json=payload)
    assert res1.status_code == 201
    body1 = res1.json()
    assert body1["status"] == "PENDING"

    analysis = (
        await db_session.execute(select(GapAnalysis).where(GapAnalysis.id == UUID(body1["id"])))
    ).scalars().first()
    analysis.status = GapAnalysisStatus.DONE
    db_session.add(make_gap_result(analysis.id))
    await db_session.commit()

    async def _fail_process(_gap_analysis_id):
        raise AssertionError("LLM should not be called for cached DONE")

    monkeypatch.setattr(llm_service, "process_gap_analysis", _fail_process)

    res2 = await client.post("/api/v1/gap-analyses", json=payload)
    assert res2.status_code == 200
    body2 = res2.json()
    assert body2["status"] == "DONE"
    assert body2["result"]["roadmap_markdown"] == "rm"
    assert body2["result"]["match_percent"] == 80.0
    assert body2["result"]["match_reason"] == "Matched 8 of 10 skills"

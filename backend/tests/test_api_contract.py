import pytest
from uuid import UUID
from sqlalchemy import select

from app.models.gap import GapAnalysis, GapAnalysisStatus
from app.services import llm_service
from .factories import GapAnalysisPayloadFactory, make_gap_result


@pytest.mark.asyncio
async def test_post_and_get_gap_analysis(client, db_session, monkeypatch):
    async def _noop(_gap_analysis_id):
        return None

    monkeypatch.setattr(llm_service, "process_gap_analysis", _noop)

    payload = GapAnalysisPayloadFactory().build()
    res = await client.post("/api/v1/gap-analyses", json=payload)
    assert res.status_code == 201
    body = res.json()
    assert "id" in body
    assert body["status"] == "PENDING"

    analysis_id = body["id"]
    analysis_uuid = UUID(analysis_id)
    analysis = (await db_session.execute(select(GapAnalysis).where(GapAnalysis.id == analysis_uuid))).scalars().first()
    analysis.status = GapAnalysisStatus.DONE
    db_session.add(make_gap_result(analysis.id))
    await db_session.commit()

    res_get = await client.get(f"/api/v1/gap-analyses/{analysis_id}")
    assert res_get.status_code == 200
    body_get = res_get.json()
    assert body_get["status"] == "DONE"
    assert body_get["result"]["roadmap_markdown"] == "rm"
    assert body_get["result"]["match_percent"] == 80.0
    assert body_get["result"]["match_reason"] == "Matched 8 of 10 skills"

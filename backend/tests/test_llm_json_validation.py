import asyncio
import pytest
from sqlalchemy import select

from app.models.gap import GapAnalysis, GapAnalysisStatus, GapResult
from app.services.gap_analysis_ai import run_gap_analysis_ai, _parse_json
from .factories import make_gap_analysis


class ValidProvider:
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


class InvalidProvider:
    name = "invalid"
    model = "invalid-1"
    last_usage = None

    def __init__(self):
        self.calls = 0

    async def generate(self, prompt: str) -> str:
        self.calls += 1
        return "not-json"


@pytest.mark.asyncio
async def test_llm_json_valid_stores_gap_results(db_session):
    analysis = make_gap_analysis()
    db_session.add(analysis)
    await db_session.commit()

    await run_gap_analysis_ai(db_session, analysis.id, ValidProvider(), prompt="x")

    row = (await db_session.execute(select(GapAnalysis).where(GapAnalysis.id == analysis.id))).scalars().first()
    assert row.status == GapAnalysisStatus.DONE
    res = (await db_session.execute(select(GapResult).where(GapResult.gap_analysis_id == analysis.id))).scalars().first()
    assert res is not None
    assert res.roadmap_markdown


@pytest.mark.asyncio
async def test_llm_json_invalid_retry_then_failed_validation(db_session):
    analysis = make_gap_analysis()
    db_session.add(analysis)
    await db_session.commit()

    provider = InvalidProvider()
    await run_gap_analysis_ai(db_session, analysis.id, provider, prompt="x")

    row = (await db_session.execute(select(GapAnalysis).where(GapAnalysis.id == analysis.id))).scalars().first()
    assert row.status == GapAnalysisStatus.FAILED_VALIDATION
    assert provider.calls == 2


def test_parse_json_with_fenced_block_and_prefix_suffix():
    raw = (
        "Here is output:\n"
        "```json\n"
        "{\"missing_skills\":[],\"action_steps\":[],\"interview_questions\":[],\"roadmap_markdown\":\"x\",\"match_percent\":0,\"match_reason\":\"m\"}\n"
        "```\n"
        "thanks"
    )
    data = _parse_json(raw)
    assert isinstance(data, dict)
    assert "missing_skills" in data


def test_parse_json_extracts_embedded_object_when_not_pure_json():
    raw = (
        "Model preface... "
        "{\"missing_skills\":[\"rust\"],\"action_steps\":[],\"interview_questions\":[],\"roadmap_markdown\":\"x\",\"match_percent\":0,\"match_reason\":\"m\"} "
        "trailing notes"
    )
    data = _parse_json(raw)
    assert data["missing_skills"] == ["rust"]

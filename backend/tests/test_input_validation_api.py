import pytest


@pytest.mark.asyncio
async def test_validate_input_rejects_short_non_tech(client):
    payload = {
        "resume_text": "Saya kerja di tim.",
        "jd_text": "Lowongan backend engineer.",
    }
    res = await client.post("/api/v1/gap-analyses/validate-input", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert body["is_valid"] is False
    assert "insufficient" in (body.get("error_message") or "").lower()


def test_validate_input_rejects_short_tech_dense_text(monkeypatch):
    from app.services import input_validation as validation

    monkeypatch.setattr(
        validation,
        "extract_skills",
        lambda _text: ["python", "sql", "docker", "kubernetes", "aws", "typescript"],
    )
    out = validation.validate_input_quality_raw("short text", "short jd")
    assert out.is_valid is False


@pytest.mark.asyncio
async def test_validate_input_accepts_long_text_even_low_tech(client):
    resume = " ".join(["engineer"] * 55)
    jd = " ".join(["requirement"] * 60)
    res = await client.post(
        "/api/v1/gap-analyses/validate-input",
        json={"resume_text": resume, "jd_text": jd},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["is_valid"] is True
    assert body["resume_word_count"] >= 50
    assert body["jd_word_count"] >= 50

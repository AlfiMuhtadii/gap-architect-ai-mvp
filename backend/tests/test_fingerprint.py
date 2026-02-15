from app.services.gap_analysis_service import _fingerprint, _normalize_text


def test_fingerprint_determinism_same_text_same_fingerprint():
    resume_a = "Senior Engineer\nPython  "
    jd_a = "Looking for PYTHON engineer"
    resume_b = " senior engineer  python"
    jd_b = "looking   for python engineer"

    fp1 = _fingerprint(_normalize_text(resume_a), _normalize_text(jd_a))
    fp2 = _fingerprint(_normalize_text(resume_b), _normalize_text(jd_b))

    assert fp1 == fp2


def test_fingerprint_determinism_different_text_different_fingerprint():
    resume_a = "Senior Engineer Python"
    jd_a = "Looking for Python engineer"
    resume_b = "Senior Engineer Java"
    jd_b = "Looking for Java engineer"

    fp1 = _fingerprint(_normalize_text(resume_a), _normalize_text(jd_a))
    fp2 = _fingerprint(_normalize_text(resume_b), _normalize_text(jd_b))

    assert fp1 != fp2

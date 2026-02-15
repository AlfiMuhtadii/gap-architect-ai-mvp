import pytest

from app.services.skill_taxonomy import extract_skills


pytestmark = pytest.mark.skip(
    reason="Template tests for upcoming entity refiner (longest-match + alias normalization)."
)


def _norm_list(values: list[str]) -> list[str]:
    return [v.strip().lower() for v in values if str(v).strip()]


def test_longest_match_wins_template():
    text = "Spring Boot (Kotlin)"
    skills = _norm_list(extract_skills(text))
    assert "spring boot" in skills
    assert "kotlin" in skills
    assert "spring" not in skills
    assert "boot" not in skills


@pytest.mark.parametrize(
    "text",
    [
        "spring-boot",
        "springboot",
        "Spring Boot",
    ],
)
def test_alias_normalization_template(text: str):
    skills = _norm_list(extract_skills(text))
    assert skills.count("spring boot") == 1


def test_punctuation_mixed_stack_template():
    text = "Node.js/TypeScript + PostgreSQL"
    skills = _norm_list(extract_skills(text))
    assert "node.js" in skills or "node js" in skills
    assert "typescript" in skills
    assert "postgresql" in skills


from app.services import skill_taxonomy


def test_short_synonym_applies_only_to_exact_token() -> None:
    assert skill_taxonomy.normalize_skill_text("js") == "javascript"
    assert skill_taxonomy.normalize_skill_text("ts") == "typescript"
    assert skill_taxonomy.normalize_skill_text("golang") == "go"

    # Must not rewrite as substring replacements.
    assert skill_taxonomy.normalize_skill_text("jsdom") == "jsdom"
    assert skill_taxonomy.normalize_skill_text("ejs") == "ejs"
    assert skill_taxonomy.normalize_skill_text("jscode") == "jscode"


def test_extract_skills_does_not_promote_substring_tokens(monkeypatch) -> None:
    monkeypatch.setattr(
        skill_taxonomy,
        "get_skill_taxonomy_map",
        lambda: {
            "javascript": "javascript",
            "ejs": "ejs",
            "jsdom": "jsdom",
        },
    )
    text = "I use ejs templates and jsdom for testing."
    skills = [s.lower() for s in skill_taxonomy.extract_skills(text)]
    assert "ejs" in skills
    assert "jsdom" in skills
    assert "javascript" not in skills


from __future__ import annotations

import re

from app.services.skill_taxonomy import extract_skills


_WORD_RE = re.compile(r"[a-z0-9\+\#\-\.]+")
_SKILL_STOPWORDS = {"on", "of", "and", "the", "for", "to", "in", "with", "as", "at"}
_SHORT_TECH_TOKENS = {"c", "r", "go", "c#", "f#"}


def _tokens(text: str) -> list[str]:
    raw = _WORD_RE.findall(text.lower())
    # Keep short but meaningful technical tokens such as c, c#, and go.
    return [t for t in raw if len(t) >= 2 or t in _SHORT_TECH_TOKENS]


def match_skills(
    resume_text: str,
    jd_text: str,
) -> tuple[list[str], float, str, list[str]]:
    taxonomy_jd = extract_skills(jd_text)
    if taxonomy_jd:
        taxonomy_resume = set(extract_skills(resume_text))
        if taxonomy_resume:
            matched = [s for s in taxonomy_jd if s in taxonomy_resume]
            missing = [s for s in taxonomy_jd if s not in taxonomy_resume]
            missing, removed = _filter_missing_by_resume_text(missing, resume_text)
            total = max(len(taxonomy_jd), 1)
            match_percent = round((1 - (len(missing) / total)) * 100, 2)
            reason = f"Skill taxonomy matched {total - len(missing)} of {total} JD skills."
            if removed:
                reason = f"{reason} Resume text filter applied."
            top = _top_priority_from_matches(matched, jd_text)
            return missing, match_percent, reason, top

        # Fallback: token overlap between JD skills and resume text
        resume_tokens = set(_tokens(resume_text))
        matched = [s for s in taxonomy_jd if _skill_in_tokens(s, resume_tokens)]
        missing = [s for s in taxonomy_jd if s not in matched]
        missing, removed = _filter_missing_by_resume_text(missing, resume_text)
        total = max(len(taxonomy_jd), 1)
        match_percent = round((1 - (len(missing) / total)) * 100, 2)
        reason = f"JD skills matched by token overlap; {total - len(missing)} of {total}."
        if removed:
            reason = f"{reason} Resume text filter applied."
        top = _top_priority_from_matches(matched, jd_text)
        return missing, match_percent, reason, top

    return _fallback_token_match(resume_text, jd_text)


def _fallback_token_match(
    resume_text: str, jd_text: str
) -> tuple[list[str], float, str, list[str]]:
    jd_skills = extract_skills(jd_text)
    resume_skills = set(extract_skills(resume_text))

    if not jd_skills:
        jd_skills = list(dict.fromkeys(_tokens(jd_text)))
    if not resume_skills:
        resume_skills = set(_tokens(resume_text))

    matched = [s for s in jd_skills if s in resume_skills]
    missing = [s for s in jd_skills if s not in resume_skills]
    missing, removed = _filter_missing_by_resume_text(missing, resume_text)
    total = max(len(jd_skills), 1)
    match_percent = round((1 - (len(missing) / total)) * 100, 2)
    reason = f"Fallback token match on {total} JD skills; {len(missing)} missing identified."
    if removed:
        reason = f"{reason} Resume text filter applied."
    top = _top_priority_from_matches(matched, jd_text)
    return missing, match_percent, reason, top


def _skill_in_tokens(skill: str, tokens: set[str]) -> bool:
    parts = [p for p in _tokens(skill) if p not in _SKILL_STOPWORDS]
    if not parts:
        return False
    return all(p in tokens for p in parts)


def _contains_skill_phrase(text: str, skill: str) -> bool:
    parts = [p for p in _tokens(skill) if p not in _SKILL_STOPWORDS]
    if not parts:
        return False
    if len(parts) == 1:
        pattern = r"\b" + re.escape(parts[0]) + r"\b"
        return re.search(pattern, text) is not None
    phrase = r"\b" + r"\s+".join(re.escape(p) for p in parts) + r"\b"
    return re.search(phrase, text) is not None


def _filter_missing_by_resume_text(missing: list[str], resume_text: str) -> tuple[list[str], int]:
    if not missing:
        return missing, 0
    resume_tokens = set(_tokens(resume_text))
    resume_text_norm = " ".join(_WORD_RE.findall(resume_text.lower()))
    filtered: list[str] = []
    removed = 0
    for s in missing:
        if _skill_in_tokens(s, resume_tokens) or _contains_skill_phrase(resume_text_norm, s):
            removed += 1
            continue
        filtered.append(s)
    return filtered, removed


def _filter_missing_by_jd_text(missing: list[str], jd_text: str) -> tuple[list[str], int]:
    if not missing:
        return missing, 0
    jd_tokens = set(_tokens(jd_text))
    jd_text_norm = " ".join(_WORD_RE.findall(jd_text.lower()))
    filtered: list[str] = []
    removed = 0
    for s in missing:
        if not _skill_in_tokens(s, jd_tokens) and not _contains_skill_phrase(jd_text_norm, s):
            removed += 1
            continue
        filtered.append(s)
    return filtered, removed


def _top_priority_from_matches(matched_skills: list[str], jd_text: str, limit: int = 5) -> list[str]:
    if not matched_skills:
        return []
    lower = jd_text.lower()
    scores: list[tuple[str, int]] = []
    for s in matched_skills:
        count = lower.count(s.lower())
        scores.append((s, count))
    scores.sort(key=lambda x: x[1], reverse=True)
    top = [s for s, c in scores if c > 0]
    return top[:limit] if top else matched_skills[:limit]




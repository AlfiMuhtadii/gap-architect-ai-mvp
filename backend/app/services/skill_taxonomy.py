from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import Iterable

from app.core.config import settings

logger = logging.getLogger("app.skill_taxonomy")


_WORD_RE = re.compile(r"[a-z0-9\+\#\-\.]+")
_STOPWORDS = {
    "and",
    "the",
    "with",
    "for",
    "to",
    "in",
    "of",
    "a",
    "on",
    "is",
    "are",
    "be",
    "as",
    "or",
    "we",
    "you",
    "our",
    "looking",
    "seeking",
    "hire",
    "hiring",
    "need",
    "needs",
    "require",
    "required",
    "requirements",
    "role",
    "position",
    "candidate",
    "candidates",
    "responsibilities",
    "responsibility",
    "about",
    "team",
    "company",
    "job",
    "description",
    "purpose",
    "maintain",
    "supporting",
    "environment",
    "key",
}

_OPEN_VOCAB_HINTS = {
    "dotnet",
    "scala",
    "kotlin",
    "akka",
    "tokio",
    "cassandra",
    "flink",
    "spring",
    "boot",
    "bun",
    "mojo",
    "svelte",
    "tailwind",
    "nestjs",
    "nextjs",
    "fastapi",
    "django",
    "flask",
    "kafka",
    "pulsar",
    "airflow",
    "spark",
    "hadoop",
    "terraform",
    "ansible",
    "graphql",
    "grpc",
    "docker",
    "kubernetes",
    "aws",
    "gcp",
    "azure",
}

_OPEN_VOCAB_BIGRAMS = {
    "spring boot",
    "akka streams",
    "dotnet core",
    "aspnet core",
}

_SHORT_CANONICAL_SKILLS = {"c", "r", "go", "c#", "f#"}
# Keep open-vocab short-token hints very strict to reduce false positives.
_SHORT_TECH_HINTS = {"go"}
_MERGE_TOKEN_BIGRAMS = {
    ("c", "sharp"): "c#",
    ("f", "sharp"): "f#",
    ("react", "js"): "reactjs",
    ("node", "js"): "nodejs",
    ("next", "js"): "nextjs",
}
_TOKEN_SYNONYMS = {
    "golang": "go",
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "k8s": "kubernetes",
    "tf": "terraform",
    "pg": "postgresql",
    "psql": "postgresql",
}


def _canonical_token(token: str) -> str:
    t = token.lower().strip()
    if not t:
        return ""
    if t in {"c#", "c++", "f#"}:
        return t
    if t.startswith(".") and len(t) > 1:
        t = "dot" + t[1:]
    t = re.sub(r"[^a-z0-9\+#]+", "", t)
    if t == "csharp":
        return "c#"
    if t in {"cplusplus", "cpp"}:
        return "c++"
    if t == "fsharp":
        return "f#"
    t = _TOKEN_SYNONYMS.get(t, t)
    return t


def _canonical_tokens(text: str) -> list[str]:
    raw = _WORD_RE.findall(text.lower())
    return [t for t in (_canonical_token(tok) for tok in raw) if t]


def _canonical_key(text: str) -> str:
    tokens = _canonical_tokens(text)
    if not tokens:
        return ""
    merged: list[str] = []
    i = 0
    while i < len(tokens):
        if i + 1 < len(tokens):
            pair = (tokens[i], tokens[i + 1])
            mapped = _MERGE_TOKEN_BIGRAMS.get(pair)
            if mapped:
                merged.append(mapped)
                i += 2
                continue
        merged.append(tokens[i])
        i += 1
    return " ".join(merged)


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def normalize_skill_text(text: str) -> str:
    return _canonical_key(text)


def _tokenize(text: str) -> list[str]:
    key = _canonical_key(text)
    return key.split() if key else []


def _looks_like_tech_token(token: str) -> bool:
    if token in _SHORT_TECH_HINTS:
        return True
    if len(token) < 2 or len(token) > 30:
        return False
    if token in _STOPWORDS:
        return False
    if token in _OPEN_VOCAB_HINTS:
        return True
    if any(ch in token for ch in "+#.-"):
        if not any(ch.isalpha() for ch in token):
            return False
        return True
    if any(ch.isdigit() for ch in token):
        if not any(ch.isalpha() for ch in token):
            return False
        return True
    suffixes = ("js", "sql", "db", "api", "sdk", "ops", "ml", "ai")
    if token.endswith(suffixes):
        return True
    return False


def _extract_open_vocab_skills(text: str, known_norm: set[str], limit: int = 40) -> list[str]:
    tokens = _tokenize(text)
    if not tokens:
        return []
    out: list[str] = []
    seen: set[str] = set()
    used_positions: set[int] = set()
    max_n = 2
    for i in range(len(tokens)):
        matched = False
        for n in range(max_n, 0, -1):
            if i + n > len(tokens):
                continue
            span = range(i, i + n)
            if any(pos in used_positions for pos in span):
                continue
            parts = tokens[i : i + n]
            if any(p in _STOPWORDS for p in parts):
                continue
            if n == 2 and " ".join(parts) not in _OPEN_VOCAB_BIGRAMS:
                continue
            if not any(_looks_like_tech_token(p) for p in parts):
                continue
            phrase = " ".join(parts)
            if phrase in known_norm or phrase in seen:
                continue
            seen.add(phrase)
            out.append(phrase)
            used_positions.update(span)
            matched = True
            if len(out) >= limit:
                return out
            break
        if matched:
            continue
    return out




def _load_canonical_skills(path: str) -> list[str]:
    skills: list[str] = []
    if not path:
        return skills
    if not os.path.exists(path):
        return skills
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            norm = _normalize(line)
            if not norm:
                continue
            canonical_norm = _canonical_key(line)
            if len(norm) < 3 and canonical_norm not in _SHORT_CANONICAL_SKILLS:
                continue
            if norm in _STOPWORDS:
                continue
            skills.append(line)
    return skills


@lru_cache(maxsize=1)
def get_skill_taxonomy() -> list[str]:
    return _load_canonical_skills(settings.canonical_skills_path)


def clear_skill_taxonomy_cache() -> None:
    get_skill_taxonomy.cache_clear()
    get_skill_taxonomy_map.cache_clear()


async def get_skill_taxonomy_db() -> list[str]:
    return []


@lru_cache(maxsize=1)
def get_skill_taxonomy_map() -> dict[str, str]:
    skills = get_skill_taxonomy()
    mapping: dict[str, str] = {}
    for s in skills:
        norm = _canonical_key(s)
        if norm:
            mapping[norm] = s
        # Alias key: remove whitespace to link "springboot" <-> "spring boot".
        compact = norm.replace(" ", "")
        if compact and compact not in mapping:
            mapping[compact] = s
    return mapping


async def get_skill_taxonomy_map_db() -> dict[str, str]:
    return {}


def extract_skills(text: str) -> list[str]:
    mapping = get_skill_taxonomy_map()
    tokens = _tokenize(text)
    if not tokens:
        return []
    max_n = 4
    skill_set = set(mapping.keys())
    found: list[str] = []
    emitted_norm: set[str] = set()
    used_positions: set[int] = set()

    # Greedy longest-match-wins to suppress overlapping sub-tokens.
    for i in range(len(tokens)):
        matched = False
        for n in range(max_n, 0, -1):
            if i + n > len(tokens):
                continue
            span = range(i, i + n)
            if any(pos in used_positions for pos in span):
                continue
            phrase_tokens = tokens[i : i + n]
            phrase = " ".join(phrase_tokens)
            compact = "".join(phrase_tokens)
            key = phrase if phrase in skill_set else compact
            if key not in skill_set:
                continue
            canonical = mapping[key]
            canonical_norm = _canonical_key(canonical)
            if canonical_norm not in emitted_norm:
                found.append(canonical)
                emitted_norm.add(canonical_norm)
            used_positions.update(span)
            matched = True
            break
        if matched:
            continue

    def _is_subphrase(candidate: str, existing: list[str]) -> bool:
        c_tokens = candidate.split()
        if not c_tokens:
            return True
        for item in existing:
            e_tokens = _canonical_key(item).split()
            if len(e_tokens) < len(c_tokens):
                continue
            if all(token in e_tokens for token in c_tokens):
                return True
        return False

    # Open-vocabulary fallback: keep tech-like skills even if not in canonical list.
    known_norm = set(emitted_norm)
    open_vocab = _extract_open_vocab_skills(text, known_norm=known_norm)
    for phrase in open_vocab:
        if phrase in known_norm:
            continue
        if _is_subphrase(phrase, found):
            continue
        found.append(phrase)
        known_norm.add(phrase)
    deduped: list[str] = []
    seen_norm: set[str] = set()
    for item in found:
        norm = _canonical_key(item)
        if not norm or norm in seen_norm:
            continue
        seen_norm.add(norm)
        deduped.append(item)

    # Longest-match suppression on final output:
    # if a skill is a strict token-subset of another detected skill, drop the shorter one.
    normalized_items = [(item, _canonical_key(item).split()) for item in deduped]
    final_items: list[str] = []
    for item, tokens in normalized_items:
        if not tokens:
            continue
        is_subset = False
        token_set = set(tokens)
        for other, other_tokens in normalized_items:
            if item == other:
                continue
            if len(other_tokens) <= len(tokens):
                continue
            if token_set.issubset(set(other_tokens)):
                is_subset = True
                break
        if not is_subset:
            final_items.append(item)
    return final_items


def extract_missing_skills(resume_text: str, jd_text: str) -> list[str]:
    resume_norm = {normalize_skill_text(s) for s in extract_skills(resume_text)}
    jd_skills = extract_skills(jd_text)
    return [s for s in jd_skills if normalize_skill_text(s) not in resume_norm]

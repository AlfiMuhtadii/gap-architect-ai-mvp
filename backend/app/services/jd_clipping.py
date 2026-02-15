from __future__ import annotations

import re

from app.services.text_processing import find_first_keyword_index, word_count


def clip_job_description(raw_jd_text: str) -> tuple[str, str]:
    """
    Smart clipping for long JD:
    - Keep short JDs unchanged.
    - Prefer requirement-like sections using multilingual headers.
    - Fall back to bullet-density heuristic.
    - Last fallback: keep tail 80% (drop likely company intro).
    """
    text = " ".join(raw_jd_text.split()).strip()
    if not text:
        return text, "empty"

    words = word_count(text)
    if words <= 1000 and len(text) <= 4000:
        return text, "full_text_short_jd"

    start_headers = [
        "requirements",
        "qualifications",
        "what you'll do",
        "what you will do",
        "technical skills",
        "responsibilities",
        "looking for",
        "technical stack",
        "who you are",
        "kualifikasi",
        "persyaratan",
        "keahlian",
        "tanggung jawab",
        "kriteria",
    ]
    end_headers = [
        "benefits",
        "about us",
        "about the company",
        "company profile",
        "culture",
        "equal opportunity",
        "tentang perusahaan",
        "keuntungan",
    ]

    start_idx = find_first_keyword_index(text, start_headers)
    if start_idx >= 0:
        clipped = text[start_idx:]
        end_idx = find_first_keyword_index(clipped, end_headers)
        if end_idx > 0:
            clipped = clipped[:end_idx]
        return clipped.strip(), "section_clipped"

    lines = [ln.strip() for ln in raw_jd_text.splitlines() if ln.strip()]
    if lines:
        bullet_pattern = re.compile(r"^([\-*\u2022]|(\d+[\.\)]))\s+")
        total = len(lines)
        first_half = lines[: max(1, total // 2)]
        first_half_bullets = sum(1 for ln in first_half if bullet_pattern.search(ln))
        density = first_half_bullets / max(len(first_half), 1)
        if density >= 0.5:
            return text, "full_text_bulleted"

    tail_start = int(len(text) * 0.2)
    return text[tail_start:].strip(), "tail_80_no_keyword"

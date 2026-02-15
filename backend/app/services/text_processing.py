from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    lowered = text.lower()
    # Keep skill-critical symbols so C#, C++, .NET, ASP.NET are preserved.
    cleaned = re.sub(r"[^a-z0-9\s\+\#\.\-]+", " ", lowered)
    return " ".join(cleaned.split()).strip()


def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def find_first_keyword_index(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    indexes = [lowered.find(k) for k in keywords if lowered.find(k) != -1]
    return min(indexes) if indexes else -1


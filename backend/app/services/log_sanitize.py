from __future__ import annotations

import re


def sanitize_for_log(value: object, *, max_len: int = 500) -> str:
    text = str(value) if value is not None else ""
    # Prevent log injection via newlines/control chars.
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[\x00-\x1f\x7f]+", " ", text)
    text = " ".join(text.split())
    if len(text) > max_len:
        return text[:max_len]
    return text


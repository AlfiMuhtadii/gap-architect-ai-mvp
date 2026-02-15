from __future__ import annotations

import hashlib
import json
import time
import asyncio
import logging
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ValidationError, field_validator, conlist, Field
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.models.gap import GapAnalysis, GapAnalysisStatus, GapResult, LlmRun, LlmRunStatus
from app.core.config import settings
from app.services.log_sanitize import sanitize_for_log
from app.services.skill_matcher import match_skills
from app.services.skill_taxonomy import extract_skills, get_skill_taxonomy_map, normalize_skill_text

logger = logging.getLogger("app.gap_analysis_ai")


class LlmProvider(Protocol):
    name: str
    model: str
    last_usage: dict[str, Any] | None

    async def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        ...


class ActionStep(BaseModel):
    title: str
    why: str
    deliverable: str

    @field_validator("title", "why", "deliverable")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-empty")
        return v


class InterviewQuestion(BaseModel):
    question: str
    focus_gap: str
    what_good_looks_like: str

    @field_validator("question", "focus_gap", "what_good_looks_like")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must be non-empty")
        return v


class GapAnalysisAIResult(BaseModel):
    jd_skills_extracted: list[str] = Field(default_factory=list)
    resume_skills_extracted: list[str] = Field(default_factory=list)
    missing_skills: list[str]
    top_priority_skills: list[str] = Field(default_factory=list)
    action_steps: conlist(ActionStep, min_length=3, max_length=3)
    interview_questions: conlist(InterviewQuestion, min_length=3, max_length=3)
    roadmap_markdown: str
    match_percent: float
    match_reason: str

    @field_validator("missing_skills")
    @classmethod
    def _skills_non_empty(cls, v: list[str]) -> list[str]:
        if any(not s.strip() for s in v):
            raise ValueError("missing_skills must contain non-empty strings")
        return v

    @field_validator("roadmap_markdown")
    @classmethod
    def _roadmap_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("roadmap_markdown must be non-empty")
        _validate_roadmap_markdown(v)
        return v

    @field_validator("match_percent")
    @classmethod
    def _match_percent_range(cls, v: float) -> float:
        if v < 0 or v > 100:
            raise ValueError("match_percent must be between 0 and 100")
        return v

    @field_validator("match_reason")
    @classmethod
    def _match_reason_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("match_reason must be non-empty")
        return v

    @field_validator("top_priority_skills")
    @classmethod
    def _list_non_null(cls, v: list[str]) -> list[str]:
        if v is None:
            raise ValueError("must be a list")
        return v


class LlmCallError(RuntimeError):
    def __init__(self, message: str, *, duration_ms: int = 0, attempts: int = 1) -> None:
        super().__init__(message)
        self.duration_ms = duration_ms
        self.attempts = attempts


def _flatten_exception_messages(exc: BaseException) -> list[str]:
    messages: list[str] = []
    seen: set[int] = set()

    def _walk(err: BaseException | None) -> None:
        if err is None:
            return
        ident = id(err)
        if ident in seen:
            return
        seen.add(ident)

        if hasattr(err, "exceptions") and isinstance(getattr(err, "exceptions"), tuple):
            for child in getattr(err, "exceptions"):
                if isinstance(child, BaseException):
                    _walk(child)

        text = str(err).strip()
        if text and text not in messages:
            messages.append(text)

        cause = getattr(err, "__cause__", None)
        if isinstance(cause, BaseException):
            _walk(cause)
        context = getattr(err, "__context__", None)
        if isinstance(context, BaseException):
            _walk(context)

    _walk(exc)
    return messages


def _exception_contains_token(exc: BaseException, token: str) -> bool:
    token_lower = token.lower()
    for msg in _flatten_exception_messages(exc):
        if token_lower in msg.lower():
            return True
    return False


def _exception_summary(exc: BaseException) -> str:
    msgs = _flatten_exception_messages(exc)
    if msgs:
        return " | ".join(msgs[:3])
    return exc.__class__.__name__


def _estimate_cost_usd(model: str, usage: dict[str, Any] | None) -> float | None:
    if not usage:
        return None
    prompt_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    if total_tokens <= 0:
        return None

    pricing_per_1m: dict[str, tuple[float, float]] = {
        "gpt-4o-mini": (0.15, 0.60),
        "gpt-4.1-mini": (0.40, 1.60),
    }
    key = model.lower()
    if key not in pricing_per_1m:
        return None
    in_price, out_price = pricing_per_1m[key]
    return round((prompt_tokens / 1_000_000 * in_price) + (completion_tokens / 1_000_000 * out_price), 8)


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _parse_json(raw: str) -> dict[str, Any]:
    if not isinstance(raw, str):
        raise ValueError("response must be a JSON string")
    text = raw.strip()
    if not text:
        raise ValueError("empty response")
    if text.startswith("```"):
        fence = text.split("```")
        if len(fence) >= 3:
            text = fence[1]
            if text.strip().lower().startswith("json"):
                text = text.strip()[4:]
            text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        extracted = _extract_first_json_object(text)
        if not extracted:
            raise
        data = json.loads(extracted)
    if isinstance(data, dict) and isinstance(data.get("raw"), str):
        return _parse_json(data["raw"])
    if not isinstance(data, dict):
        raise ValueError("response must be a JSON object")
    return data


def _extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _fallback_roadmap_markdown(missing_skills: list[str], steps: list[dict[str, str]]) -> str:
    missing = [s for s in missing_skills if s.strip()]
    def _step_fields(step: Any) -> dict[str, str]:
        if isinstance(step, dict):
            return {
                "title": str(step.get("title", "")).strip(),
                "why": str(step.get("why", "")).strip(),
                "deliverable": str(step.get("deliverable", "")).strip(),
            }
        if hasattr(step, "title") and hasattr(step, "why") and hasattr(step, "deliverable"):
            return {
                "title": str(getattr(step, "title", "")).strip(),
                "why": str(getattr(step, "why", "")).strip(),
                "deliverable": str(getattr(step, "deliverable", "")).strip(),
            }
        return {"title": "", "why": "", "deliverable": ""}

    steps_used = [_step_fields(s) for s in steps[:3] if s]
    gap_summary = (
        "The resume covers the core requirements, with only minor gaps."
        if not missing
        else (
            "The resume aligns with the role, but several JD skills are missing. "
            f"Priority gaps include {missing[0]}"
            + (f" and {missing[1]}." if len(missing) > 1 else ".")
        )
    )
    priority = "\n".join([f"- {s}" for s in missing[:3]]) if missing else "- None"
    steps_md = (
        "\n".join(
            [
                f"### Step {i+1}  {step['title']}\n**Why:** {step['why']}\n**Deliverable:** {step['deliverable']}"
                for i, step in enumerate(steps_used)
            ]
        )
        if steps_used
        else "No concrete steps generated."
    )
    outcomes = (
        "\n".join(
            [
                f"- Demonstrates applied skill in {missing[0]}.",
                f"- Produces reviewable evidence for {missing[1] if len(missing) > 1 else missing[0]}.",
                "- Shows readiness for senior-level implementation work.",
            ]
        )
        if missing
        else "- No outcomes generated."
    )
    learning_order = (
        "\n".join([f"{i+1}. {s}" for i, s in enumerate(missing[:3])]) if missing else "1. None"
    )
    return (
        "## Gap Summary\n"
        f"{gap_summary}\n\n## Priority Skills to Learn\n"
        f"{priority}\n\n## Concrete Steps\n"
        f"{steps_md}\n\n## Expected Outcomes / Readiness\n"
        f"{outcomes}\n\n## Suggested Learning Order\n"
        f"{learning_order}"
    )


def _inject_steps_into_roadmap(roadmap: str, steps: list[dict[str, str]]) -> str:
    import re

    if not roadmap:
        return roadmap
    if not steps:
        return roadmap

    steps_md = "\n".join(
        [
            f"### Step {idx+1}  {step.get('title','')}\n**Why:** {step.get('why','')}\n**Deliverable:** {step.get('deliverable','')}"
            for idx, step in enumerate(steps)
        ]
    )

    pattern = r"(## Concrete Steps\s*)(.*?)(\n## |\Z)"
    match = re.search(pattern, roadmap, flags=re.S)
    if not match:
        return roadmap
    prefix = match.group(1)
    suffix = match.group(3)
    return re.sub(pattern, f"{prefix}{steps_md}\n{suffix}", roadmap, flags=re.S)

def _ensure_three_topics(missing_skills: list[str]) -> list[str]:
    base = [str(s).strip() for s in missing_skills if str(s).strip()]
    fillers = ["system design", "testing strategy", "performance optimization"]
    out: list[str] = []
    for item in base + fillers:
        if item.lower() in {x.lower() for x in out}:
            continue
        out.append(item)
        if len(out) == 3:
            break
    return out


def _fallback_action_steps(missing_skills: list[str]) -> list[dict[str, str]]:
    topics = _ensure_three_topics(missing_skills)
    return [
        {
            "title": f"Build a production-ready project using {topic}",
            "why": f"Shows practical capability in {topic} for job-ready delivery.",
            "deliverable": f"A deployable project module with tests and short architecture notes for {topic}.",
        }
        for topic in topics
    ]


def _fallback_interview_questions(missing_skills: list[str]) -> list[dict[str, str]]:
    topics = _ensure_three_topics(missing_skills)
    return [
        {
            "question": f"How would you implement {topic} in a production system?",
            "focus_gap": topic,
            "what_good_looks_like": f"Clear design trade-offs, implementation approach, and measurable outcomes for {topic}.",
        }
        for topic in topics
    ]


def _coerce_local_payload(parsed: dict[str, Any]) -> None:
    def _to_str_list(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned = []
        for item in value:
            s = str(item).strip()
            if s:
                cleaned.append(s)
        return cleaned

    missing = _to_str_list(parsed.get("missing_skills"))
    parsed["missing_skills"] = missing

    def _ensure_list(name: str) -> None:
        parsed[name] = _to_str_list(parsed.get(name))

    _ensure_list("top_priority_skills")
    _ensure_list("jd_skills_extracted")
    _ensure_list("resume_skills_extracted")

    steps = parsed.get("action_steps")
    if not isinstance(steps, list):
        parsed["action_steps"] = _fallback_action_steps(missing)
    else:
        fixed_steps = []
        for step in steps[:3]:
            title = str(step.get("title", "")).strip()
            why = str(step.get("why", "")).strip()
            deliverable = str(step.get("deliverable", "")).strip()
            if not title or not why or not deliverable:
                fixed_steps = _fallback_action_steps(missing)
                break
            fixed_steps.append({"title": title, "why": why, "deliverable": deliverable})
        if fixed_steps:
            parsed["action_steps"] = fixed_steps
    if len(parsed.get("action_steps") or []) != 3:
        parsed["action_steps"] = _fallback_action_steps(missing)

    questions = parsed.get("interview_questions")
    if not isinstance(questions, list):
        parsed["interview_questions"] = _fallback_interview_questions(missing)
    else:
        fixed_questions = []
        for q in questions[:3]:
            question = str(q.get("question", "")).strip()
            focus = str(q.get("focus_gap", "")).strip()
            good = str(q.get("what_good_looks_like", "")).strip()
            if not question or not focus or not good:
                fixed_questions = _fallback_interview_questions(missing)
                break
            fixed_questions.append(
                {"question": question, "focus_gap": focus, "what_good_looks_like": good}
            )
        if fixed_questions:
            parsed["interview_questions"] = fixed_questions
    if len(parsed.get("interview_questions") or []) != 3:
        parsed["interview_questions"] = _fallback_interview_questions(missing)

    if not isinstance(parsed.get("match_percent"), (int, float)):
        parsed["match_percent"] = 0.0
    if not isinstance(parsed.get("match_reason"), str) or not parsed.get("match_reason"):
        parsed["match_reason"] = "Local LLM response normalized."
    if not parsed.get("top_priority_skills"):
        parsed["top_priority_skills"] = [s for s in missing[:3]]
    if parsed.get("top_priority_skills"):
        parsed["top_priority_skills"] = _to_str_list(parsed.get("top_priority_skills"))[:3]


async def _build_validation_fallback_message(
    analysis: GapAnalysis,
    *,
    failure_case: str,
) -> str:
    try:
        missing, *_ = match_skills(
            resume_text=analysis.resume_text,
            jd_text=analysis.jd_text,
        )
    except Exception:  # noqa: BLE001
        missing = []
    sample = [s for s in (missing or []) if str(s).strip()][:3]
    case_messages = {
        "invalid_json_after_retry": (
            "The AI returned malformed JSON twice, so the analysis result could not be parsed."
        ),
        "schema_validation_failed_after_retry": (
            "The AI returned an output that did not match the required schema after one repair attempt."
        ),
        "schema_validation_failed": (
            "The AI returned an output that did not match the required schema."
        ),
    }
    lead = case_messages.get(
        failure_case,
        "The analysis response failed validation.",
    )
    if sample:
        return (
            f"{lead} Interim deterministic signal suggests missing skills: "
            + ", ".join(sample)
            + ". Please retry in a moment."
        )
    return f"{lead} Please retry in a moment."


def _validate_roadmap_markdown(text: str) -> None:
    import re

    # Normalize common heading variants from LLMs
    if "# Gap Summary" in text and "## Gap Summary" not in text:
        text = text.replace("# Gap Summary", "## Gap Summary", 1)
    if "# Priority Skills to Learn" in text and "## Priority Skills to Learn" not in text:
        text = text.replace("# Priority Skills to Learn", "## Priority Skills to Learn", 1)
    if "# Concrete Steps" in text and "## Concrete Steps" not in text:
        text = text.replace("# Concrete Steps", "## Concrete Steps", 1)
    if "# Expected Outcomes / Readiness" in text and "## Expected Outcomes / Readiness" not in text:
        text = text.replace("# Expected Outcomes / Readiness", "## Expected Outcomes / Readiness", 1)
    if "# Suggested Learning Order" in text and "## Suggested Learning Order" not in text:
        text = text.replace("# Suggested Learning Order", "## Suggested Learning Order", 1)

    required_headings = [
        "## Gap Summary",
        "## Priority Skills to Learn",
        "## Concrete Steps",
        "## Expected Outcomes / Readiness",
        "## Suggested Learning Order",
    ]
    positions: list[int] = []
    for heading in required_headings:
        idx = text.find(heading)
        if idx == -1:
            raise ValueError(f"roadmap_markdown missing heading: {heading}")
        positions.append(idx)
    if positions != sorted(positions):
        raise ValueError("roadmap_markdown headings out of order")

    def _section_slice(start: str, end: str | None) -> str:
        s = text.split(start, 1)[1]
        if end:
            s = s.split(end, 1)[0]
        return s.strip()

    gap_summary = _section_slice("## Gap Summary", "## Priority Skills to Learn")
    sentences = [s for s in re.split(r"[.!?]+", gap_summary) if s.strip()]
    if len(sentences) < 1:
        raise ValueError("roadmap_markdown Gap Summary must have at least 1 sentence")

    priority = _section_slice("## Priority Skills to Learn", "## Concrete Steps")
    priority_items = [
        line
        for line in priority.splitlines()
        if line.strip().startswith(("- ", "* ", "+ "))
    ]
    if not (1 <= len(priority_items) <= 10):
        raise ValueError("roadmap_markdown Priority Skills must have 1-10 bullets")

    steps = _section_slice("## Concrete Steps", "## Expected Outcomes / Readiness")
    step_titles = re.findall(r"^### Step [1-3][\s:]+.+$", steps, flags=re.M)
    if len(step_titles) < 1 or len(step_titles) > 3:
        raise ValueError("roadmap_markdown Concrete Steps must have 1-3 steps")
    if steps.count("**Why:**") < len(step_titles) or steps.count("**Deliverable:**") < len(step_titles):
        raise ValueError("roadmap_markdown Concrete Steps must include Why and Deliverable")

    outcomes = _section_slice("## Expected Outcomes / Readiness", "## Suggested Learning Order")
    outcome_items = [
        line
        for line in outcomes.splitlines()
        if line.strip().startswith(("- ", "* ", "+ "))
    ]
    if len(outcome_items) < 1:
        raise ValueError("roadmap_markdown Expected Outcomes must have at least 1 bullet")

    order = _section_slice("## Suggested Learning Order", None)
    order_items = [line for line in order.splitlines() if re.match(r"^[1-3]\.\s+", line.strip())]
    if len(order_items) < 1 or len(order_items) > 3:
        raise ValueError("roadmap_markdown Suggested Learning Order must have 1-3 items")


def _repair_prompt(base_prompt: str, raw_response: str, error: str) -> str:
    return (
        f"{base_prompt}\n\n"
        "The previous response did not match the required JSON schema.\n"
        f"Validation error: {error}\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{"
        "\"jd_skills_extracted\": [\"string\"], "
        "\"resume_skills_extracted\": [\"string\"], "
        "\"missing_skills\": [\"string\"], "
        "\"top_priority_skills\": [\"string\"], "
        "\"action_steps\": [{\"title\":\"\",\"why\":\"\",\"deliverable\":\"\"}, ... x3], "
        "\"interview_questions\": [{\"question\":\"\",\"focus_gap\":\"\",\"what_good_looks_like\":\"\"}, ... x3], "
        "\"roadmap_markdown\": \"string\", "
        "\"match_percent\": 0, "
        "\"match_reason\": \"string\""
        "}\n\n"
        "Here is the previous response to repair:\n"
        f"{raw_response}"
    )


def _infer_context(jd_text: str) -> str:
    jd = jd_text.lower()
    tags: list[str] = []
    if "saas" in jd:
        tags.append("SaaS")
    if "erp" in jd:
        tags.append("ERP")
    if "microservice" in jd or "microservices" in jd:
        tags.append("microservices")
    if "api" in jd or "grpc" in jd or "rest" in jd:
        tags.append("API services")
    if "data" in jd:
        tags.append("data workflows")
    if "cloud" in jd or "aws" in jd or "gcp" in jd or "azure" in jd:
        tags.append("cloud deployment")
    if tags:
        return " and ".join(tags[:3])
    return "backend systems"


def _build_action_steps(missing_skills: list[str], jd_text: str) -> list[dict[str, str]]:
    context = _infer_context(jd_text)
    steps: list[dict[str, str]] = []
    for skill in missing_skills[:3]:
        steps.append(
            {
                "title": f"Build a {context} module using {skill}",
                "why": f"Demonstrates practical use of {skill} in a {context} context.",
                "deliverable": f"A small, reviewable project or case study showing {skill} applied to {context}.",
            }
        )
    return steps


def _build_interview_questions(missing_skills: list[str], jd_text: str) -> list[dict[str, str]]:
    context = _infer_context(jd_text)
    questions: list[dict[str, str]] = []
    for skill in missing_skills[:3]:
        questions.append(
            {
                "question": f"Explain how you would apply {skill} in a {context} system.",
                "focus_gap": skill,
                "what_good_looks_like": f"Clear tradeoffs, implementation details, and impact using {skill}.",
            }
        )
    return questions


async def _pre_rank_missing_skills(
    session: AsyncSession, missing_skills: list[str], jd_text: str
) -> list[str]:
    if not missing_skills:
        return []
    scored: list[tuple[int, int, int, str]] = []
    jd = jd_text.lower()
    for idx, skill in enumerate(missing_skills):
        s = str(skill).strip()
        if not s:
            continue
        s_lower = s.lower()
        freq = jd.count(s_lower)
        first_pos = jd.find(s_lower)
        if first_pos < 0:
            first_pos = 10**9
        scored.append((freq, first_pos, idx, s))
    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    return [s for _, _, _, s in scored]


async def _rank_missing_skills(
    session: AsyncSession, missing_skills: list[str], jd_text: str
) -> list[str]:
    pre_ranked = await _pre_rank_missing_skills(session, missing_skills, jd_text)
    return pre_ranked[:3]



async def run_gap_analysis_ai(
    session: AsyncSession,
    gap_analysis_id: UUID,
    provider: LlmProvider,
    prompt: str,
    *,
    fallback_from: str | None = None,
    raise_on_provider_failure: bool = False,
) -> None:
    try:
        analysis = (await session.execute(select(GapAnalysis).where(GapAnalysis.id == gap_analysis_id))).scalars().first()
    except SQLAlchemyError as e:
        logger.error(
            f"Database error fetching gap_analysis {gap_analysis_id}",
            extra={"gap_analysis_id": str(gap_analysis_id), "error": sanitize_for_log(e)},
            exc_info=True,
        )
        raise ValueError(f"Database error: {e}") from e
    
    if analysis is None:
        logger.warning(f"Gap analysis {gap_analysis_id} not found")
        raise ValueError("gap_analysis not found")

    def _provider_timeout() -> float:
        return (
            settings.local_llm_timeout_seconds
            if getattr(provider, "name", "") == "local_llm"
            else settings.llm_timeout_seconds
        )

    async def _call_llm(
        call_prompt: str,
        *,
        temperature: float | None = None,
    ) -> tuple[dict[str, Any] | None, str | None, int, int]:
        start = time.perf_counter()
        attempts = 0
        while True:
            attempts += 1
            try:
                try:
                    gen_coro = provider.generate(call_prompt, temperature=temperature)
                except TypeError:
                    gen_coro = provider.generate(call_prompt)
                raw = await asyncio.wait_for(gen_coro, timeout=_provider_timeout())
                break
            except Exception as exc:  # noqa: BLE001
                is_timeout = _exception_contains_token(exc, "timeout")
                if attempts < 2 and is_timeout:
                    continue
                duration_ms = int((time.perf_counter() - start) * 1000)
                error_text = _exception_summary(exc)
                raise LlmCallError(error_text, duration_ms=duration_ms, attempts=attempts) from exc

        duration_ms = int((time.perf_counter() - start) * 1000)
        try:
            parsed = _parse_json(raw)
        except Exception as exc:  # noqa: BLE001
            error_text = _exception_summary(exc)
            return None, error_text, duration_ms, attempts

        return parsed, None, duration_ms, attempts

    try:
        parsed, error, duration_ms, attempts = await _call_llm(prompt)
        if parsed is None:
            if getattr(provider, "name", "") == "local_llm":
                raise ValueError("local_invalid_json")
            repaired_prompt = _repair_prompt(prompt, "", error or "invalid JSON")
            parsed, error, duration_ms, attempts = await _call_llm(repaired_prompt, temperature=0.2)
            if parsed is None:
                meta = {
                    "retry_attempts": attempts,
                    "fallback_from": fallback_from,
                }
                await _log_llm_run(
                    session=session,
                    gap_analysis_id=gap_analysis_id,
                    provider=provider.name,
                    model=provider.model,
                    request_hash=_hash_prompt(repaired_prompt),
                    response_json={"error": error or "invalid JSON after repair", "_meta": meta},
                    status=LlmRunStatus.FAILED,
                    error_message=error or "invalid JSON after repair",
                    duration_ms=duration_ms,
                )
                friendly = await _build_validation_fallback_message(
                    analysis,
                    failure_case="invalid_json_after_retry",
                )
                await _mark_failed_validation(session, analysis, friendly)
                return

        if getattr(provider, "name", "") == "local_llm":
            _coerce_local_payload(parsed)
            roadmap = parsed.get("roadmap_markdown") if isinstance(parsed, dict) else None
            try:
                if isinstance(roadmap, str):
                    _validate_roadmap_markdown(roadmap)
                else:
                    raise ValueError("roadmap missing")
            except Exception:
                parsed["roadmap_markdown"] = _fallback_roadmap_markdown(
                    parsed.get("missing_skills") or [],
                    parsed.get("action_steps") or [],
                )

        try:
            validated = GapAnalysisAIResult.model_validate(parsed)
        except ValidationError as exc:
            if getattr(provider, "name", "") == "local_llm":
                _coerce_local_payload(parsed)
                try:
                    validated = GapAnalysisAIResult.model_validate(parsed)
                except ValidationError as exc_local:
                    raise ValueError("local_invalid_json") from exc_local
            repaired_prompt = _repair_prompt(prompt, json.dumps(parsed, ensure_ascii=True), str(exc))
            parsed2, error2, duration_ms, attempts = await _call_llm(repaired_prompt, temperature=0.2)
            if parsed2 is None:
                meta = {
                    "retry_attempts": attempts,
                    "fallback_from": fallback_from,
                }
                await _log_llm_run(
                    session=session,
                    gap_analysis_id=gap_analysis_id,
                    provider=provider.name,
                    model=provider.model,
                    request_hash=_hash_prompt(repaired_prompt),
                    response_json={"error": error2 or "invalid JSON after repair", "_meta": meta},
                    status=LlmRunStatus.FAILED,
                    error_message=error2 or "invalid JSON after repair",
                    duration_ms=duration_ms,
                )
                friendly = await _build_validation_fallback_message(
                    analysis,
                    failure_case="invalid_json_after_retry",
                )
                await _mark_failed_validation(session, analysis, friendly)
                return
            try:
                validated = GapAnalysisAIResult.model_validate(parsed2)
                parsed = parsed2
            except ValidationError as exc2:
                meta = {
                    "retry_attempts": attempts,
                    "fallback_from": fallback_from,
                }
                await _log_llm_run(
                    session=session,
                    gap_analysis_id=gap_analysis_id,
                    provider=provider.name,
                    model=provider.model,
                    request_hash=_hash_prompt(repaired_prompt),
                    response_json={
                        "error": _exception_summary(exc2) or "schema_validation_failed_after_repair",
                        "_meta": meta,
                    },
                    status=LlmRunStatus.FAILED,
                    error_message=_exception_summary(exc2) or "schema_validation_failed_after_repair",
                    duration_ms=duration_ms,
                )
                friendly = await _build_validation_fallback_message(
                    analysis,
                    failure_case="schema_validation_failed_after_retry",
                )
                await _mark_failed_validation(session, analysis, friendly)
                return
        usage = getattr(provider, "last_usage", None) or {}
        meta = {
            "retry_attempts": attempts,
            "fallback_from": fallback_from,
            "usage": usage if isinstance(usage, dict) else {},
            "estimated_cost_usd": _estimate_cost_usd(provider.model, usage if isinstance(usage, dict) else None),
        }
        parsed_for_log = dict(parsed)
        parsed_for_log["_meta"] = meta
        try:
            persisted = await _persist_success(session, analysis, validated, provider)
            if not persisted:
                await session.rollback()
                return
            await _log_llm_run(
                session=session,
                gap_analysis_id=gap_analysis_id,
                provider=provider.name,
                model=provider.model,
                request_hash=_hash_prompt(prompt),
                response_json=parsed_for_log,
                status=LlmRunStatus.SUCCESS,
                error_message=None,
                duration_ms=duration_ms,
            )
            await session.commit()
        except Exception as exc:  # noqa: BLE001
            await session.rollback()
            await session.refresh(analysis)
            if analysis.status == GapAnalysisStatus.DONE:
                return
            analysis.status = GapAnalysisStatus.FAILED_LLM
            analysis.error_message = sanitize_for_log(exc, max_len=1000)
            await session.commit()
            return
    except LlmCallError as exc:
        meta = {
            "retry_attempts": max(getattr(exc, "attempts", 1), 1),
            "fallback_from": fallback_from,
        }
        await _log_llm_run(
            session=session,
            gap_analysis_id=gap_analysis_id,
            provider=provider.name,
            model=provider.model,
            request_hash=_hash_prompt(prompt),
            response_json={"error": str(exc) or "llm_failed", "_meta": meta},
            status=LlmRunStatus.FAILED,
            error_message=str(exc) or "llm_failed",
            duration_ms=max(getattr(exc, "duration_ms", 0), 0),
        )
        # Persist failed layer run before bubbling to fallback chain.
        await session.commit()
        if "rate_limited" in str(exc):
            raise ValueError("rate_limited") from exc
        if getattr(provider, "name", "") == "local_llm":
            if "timeout" in str(exc).lower():
                raise ValueError("local_timeout") from exc
            raise ValueError("local_llm_failed") from exc
        if getattr(provider, "name", "") != "heuristic" and raise_on_provider_failure:
            raise ValueError("llm_failed") from exc
        await session.refresh(analysis)
        if analysis.status == GapAnalysisStatus.DONE:
            return
        analysis.status = GapAnalysisStatus.FAILED_LLM
        analysis.error_message = sanitize_for_log(exc, max_len=1000)
        await session.commit()


async def _persist_success(
    session: AsyncSession,
    analysis: GapAnalysis,
    result: GapAnalysisAIResult,
    provider: LlmProvider,
) -> bool:
    def _canonicalize_skills(skills: list[str]) -> list[str]:
        mapping = get_skill_taxonomy_map()
        out: list[str] = []
        seen: set[str] = set()
        for raw in skills:
            text = str(raw).strip()
            if not text:
                continue
            norm = normalize_skill_text(text)
            if not norm:
                continue
            canonical = mapping.get(norm) or mapping.get(norm.replace(" ", "")) or text
            canonical_norm = normalize_skill_text(canonical)
            if not canonical_norm or canonical_norm in seen:
                continue
            seen.add(canonical_norm)
            out.append(canonical)
        return out

    def _normalize_source_text(text: str) -> str:
        # Use the same canonical normalizer as skill keys so hyphen/dot variants
        # (e.g. event-driven, domain-driven, node.js) are matched consistently.
        return normalize_skill_text(text)

    def _contains_phrase(source_norm: str, phrase_norm: str) -> bool:
        phrase = phrase_norm.strip()
        if not phrase:
            return False
        hay = f" {source_norm} "
        needle = f" {phrase} "
        return needle in hay

    def _verify_ai_skills(skills: list[str], source_text: str) -> list[str]:
        """
        AI propose -> deterministic verify:
        - canonical must exist in taxonomy
        - candidate must have literal evidence in source text
        """
        mapping = get_skill_taxonomy_map()
        source_norm = _normalize_source_text(source_text)
        out: list[str] = []
        seen: set[str] = set()

        for raw in skills:
            text = str(raw).strip()
            if not text:
                continue
            raw_norm = normalize_skill_text(text)
            if not raw_norm:
                continue
            canonical = mapping.get(raw_norm) or mapping.get(raw_norm.replace(" ", ""))
            if not canonical:
                continue
            canonical_norm = normalize_skill_text(canonical)
            if not canonical_norm:
                continue

            # Accept only if the original mention OR canonical phrase exists in source.
            if not (_contains_phrase(source_norm, raw_norm) or _contains_phrase(source_norm, canonical_norm)):
                continue

            if canonical_norm in seen:
                continue
            seen.add(canonical_norm)
            out.append(canonical)
        return out

    def _collapse_overlapping(skills: list[str]) -> list[str]:
        """
        Remove broader/sub-skill duplicates when a more specific phrase exists.
        Example: ruby + rails + ruby on rails -> keep ruby on rails.
        """
        deduped: list[str] = []
        seen_norm: set[str] = set()
        for item in skills:
            norm = normalize_skill_text(item)
            if not norm or norm in seen_norm:
                continue
            seen_norm.add(norm)
            deduped.append(item)

        normalized_items = [(item, normalize_skill_text(item).split()) for item in deduped]
        final_items: list[str] = []
        for item, tokens in normalized_items:
            if not tokens:
                continue
            token_set = set(tokens)
            is_subset = False
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

    # Hybrid source:
    # - deterministic baseline guarantees explicit-text recall
    # - AI extraction (verified) adds compliant semantic recall
    jd_det = _canonicalize_skills(extract_skills(analysis.jd_text))
    resume_det = _canonicalize_skills(extract_skills(analysis.resume_text))
    jd_ai_verified = _verify_ai_skills(result.jd_skills_extracted, analysis.jd_text)
    resume_ai_verified = _verify_ai_skills(result.resume_skills_extracted, analysis.resume_text)

    final_jd = _collapse_overlapping(jd_det + jd_ai_verified)
    final_resume = _collapse_overlapping(resume_det + resume_ai_verified)
    no_requirements_mode = len(final_jd) == 0

    if not no_requirements_mode:
        resume_norm = {normalize_skill_text(s) for s in final_resume}
        missing = [s for s in final_jd if normalize_skill_text(s) not in resume_norm]
        total = max(len(final_jd), 1)
        matched = max(total - len(missing), 0)
        match_percent = round((matched / total) * 100, 2)
        match_reason = f"Deterministic diff from normalized skill sets: matched {matched} of {total} JD skills."
        top = missing[:3]
    else:
        missing = []
        top = []
        match_percent = 100.0
        match_reason = "No identifiable technical requirements found in JD; gap analysis guidance is skipped."
    use_matcher = getattr(provider, "name", "") == "heuristic"
    if not no_requirements_mode and getattr(provider, "name", "") != "heuristic":
        # Keep only priority skills that exist in current missing set.
        missing_norm = {normalize_skill_text(s) for s in missing}
        filtered_top = [
            s for s in (result.top_priority_skills or [])
            if normalize_skill_text(str(s)) in missing_norm
        ]
        top = filtered_top[:3] if filtered_top else top

    if no_requirements_mode:
        action_steps = []
        interview_questions = []
    elif use_matcher:
        ranked = await _rank_missing_skills(session, missing, analysis.jd_text)
        missing_for_steps = ranked if ranked else (missing[:3] if missing else [])

        action_steps = _build_action_steps(missing_for_steps, analysis.jd_text)
        interview_questions = _build_interview_questions(missing_for_steps, analysis.jd_text)
    else:
        action_steps = result.action_steps if isinstance(result.action_steps, list) else []
        interview_questions = result.interview_questions

    if not no_requirements_mode and not action_steps:
        action_steps = _fallback_action_steps(missing)

    def _steps_to_dicts(steps: list[Any]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for step in steps:
            if isinstance(step, dict):
                out.append(
                    {
                        "title": str(step.get("title", "")).strip(),
                        "why": str(step.get("why", "")).strip(),
                        "deliverable": str(step.get("deliverable", "")).strip(),
                    }
                )
            elif hasattr(step, "model_dump"):
                data = step.model_dump()
                out.append(
                    {
                        "title": str(data.get("title", "")).strip(),
                        "why": str(data.get("why", "")).strip(),
                        "deliverable": str(data.get("deliverable", "")).strip(),
                    }
                )
        return out

    def _questions_to_dicts(questions: list[Any]) -> list[dict[str, str]]:
        out: list[dict[str, str]] = []
        for q in questions:
            if isinstance(q, dict):
                out.append(
                    {
                        "question": str(q.get("question", "")).strip(),
                        "focus_gap": str(q.get("focus_gap", "")).strip(),
                        "what_good_looks_like": str(q.get("what_good_looks_like", "")).strip(),
                    }
                )
            elif hasattr(q, "model_dump"):
                data = q.model_dump()
                out.append(
                    {
                        "question": str(data.get("question", "")).strip(),
                        "focus_gap": str(data.get("focus_gap", "")).strip(),
                        "what_good_looks_like": str(data.get("what_good_looks_like", "")).strip(),
                    }
                )
        return out

    action_steps = _steps_to_dicts(action_steps)
    interview_questions = _questions_to_dicts(interview_questions)

    if not no_requirements_mode and len(action_steps) != 3:
        action_steps = _fallback_action_steps(missing)
    if not no_requirements_mode and len(interview_questions) != 3:
        interview_questions = _fallback_interview_questions(missing)

    def _step_to_dict(step: Any) -> dict[str, str]:
        if isinstance(step, dict):
            return {
                "title": str(step.get("title", "")).strip(),
                "why": str(step.get("why", "")).strip(),
                "deliverable": str(step.get("deliverable", "")).strip(),
            }
        if hasattr(step, "title") and hasattr(step, "why") and hasattr(step, "deliverable"):
            return {
                "title": str(getattr(step, "title", "")).strip(),
                "why": str(getattr(step, "why", "")).strip(),
                "deliverable": str(getattr(step, "deliverable", "")).strip(),
            }
        return {"title": "", "why": "", "deliverable": ""}

    if no_requirements_mode:
        roadmap_markdown = (
            "## Gap Summary\n"
            "Job description does not contain identifiable technical requirements.\n\n"
            "## Priority Skills to Learn\n"
            "- None\n\n"
            "## Concrete Steps\n"
            "No guidance generated because no technical requirements were detected.\n\n"
            "## Expected Outcomes / Readiness\n"
            "- Provide a JD with explicit technical requirements for precise gap analysis.\n\n"
            "## Suggested Learning Order\n"
            "1. None"
        )
    else:
        generated_roadmap = "\n".join(
            [
                f"{i+1}. **{_step_to_dict(step)['title']}**\n   - Why: {_step_to_dict(step)['why']}\n   - Deliverable: {_step_to_dict(step)['deliverable']}"
                for i, step in enumerate(action_steps)
            ]
        )
        roadmap_markdown = result.roadmap_markdown
        if not roadmap_markdown:
            roadmap_markdown = generated_roadmap
        else:
            if len(action_steps) >= 1:
                roadmap_markdown = _inject_steps_into_roadmap(roadmap_markdown, action_steps)

    gap_result = GapResult(
        gap_analysis_id=analysis.id,
        missing_skills=missing,
        action_steps=action_steps,
        interview_questions=interview_questions,
        roadmap_markdown=roadmap_markdown,
        match_percent=match_percent,
        match_reason=match_reason,
        top_priority_skills=top,
    )

    # CAS: only one worker may transition PENDING -> DONE and persist success artifacts.
    transition = await session.execute(
        update(GapAnalysis)
        .where(
            GapAnalysis.id == analysis.id,
            GapAnalysis.status == GapAnalysisStatus.PENDING,
        )
        .values(
            status=GapAnalysisStatus.DONE,
            error_message=None,
        )
    )
    rowcount = transition.rowcount
    # Some dialects (notably SQLite in async mode) may report -1 for unknown rowcount.
    # Treat only explicit 0 as "not claimed".
    if rowcount is not None and int(rowcount) == 0:
        return False

    session.add(gap_result)
    return True


async def _mark_failed_validation(session: AsyncSession, analysis: GapAnalysis, error_message: str) -> None:
    safe_message = sanitize_for_log(error_message, max_len=1000)
    original_length = len(error_message)
    truncated_message = safe_message
    if original_length > 1000:
        logger.warning(
            f"Error message truncated for gap_analysis {analysis.id}",
            extra={
                "gap_analysis_id": str(analysis.id),
                "original_length": original_length,
                "truncated_length": len(truncated_message),
            },
        )
    await session.execute(
        update(GapAnalysis)
        .where(GapAnalysis.id == analysis.id, GapAnalysis.status != GapAnalysisStatus.DONE)
        .values(
            status=GapAnalysisStatus.FAILED_VALIDATION,
            error_message=truncated_message,
        )
    )
    await session.commit()


async def _log_llm_run(
    *,
    session: AsyncSession,
    gap_analysis_id: UUID,
    provider: str,
    model: str,
    request_hash: str,
    response_json: dict[str, Any],
    status: LlmRunStatus,
    error_message: str | None,
    duration_ms: int,
) -> None:
    run = LlmRun(
        gap_analysis_id=gap_analysis_id,
        provider=provider,
        model=model,
        request_hash=request_hash,
        response_json=response_json,
        status=status,
        error_message=error_message,
        duration_ms=duration_ms,
    )
    session.add(run)

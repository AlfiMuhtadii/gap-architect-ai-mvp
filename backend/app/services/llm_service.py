from __future__ import annotations

import json
import re
import logging
import asyncio
from typing import Iterable
import httpx
from sqlalchemy import select

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models.gap import GapAnalysis
from app.services.gap_analysis_ai import run_gap_analysis_ai, LlmProvider
from app.services.log_sanitize import sanitize_for_log
from app.services.skill_matcher import match_skills
from app.services.skill_taxonomy import extract_skills

_INFLIGHT_IDS: set[str] = set()
_INFLIGHT_GUARD = asyncio.Lock()
_JOB_SEMAPHORE = asyncio.Semaphore(max(int(settings.max_concurrent_gap_jobs), 1))


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


def _format_exception(prefix: str, exc: BaseException) -> str:
    parts = _flatten_exception_messages(exc)
    if parts:
        return f"{prefix}: {' | '.join(parts[:3])}"
    return f"{prefix}: {exc.__class__.__name__}"


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


def _rank_missing_skills(missing: list[str], jd_text: str) -> list[str]:
    if not missing:
        return []
    scored: list[tuple[int, int, int, str]] = []
    jd = jd_text.lower()
    for idx, skill in enumerate(missing):
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
    return [s for _, _, _, s in scored][:3]


def _build_action_steps(missing: list[str], jd_text: str) -> list[dict[str, str]]:
    context = _infer_context(jd_text)
    steps: list[dict[str, str]] = []
    for skill in missing[:3]:
        steps.append(
            {
                "title": f"Build a {context} module using {skill}",
                "why": f"Demonstrates practical use of {skill} in a {context} context.",
                "deliverable": f"A small, reviewable project or case study showing {skill} applied to {context}.",
            }
        )
    return steps


def _build_interview_questions(missing: list[str], jd_text: str) -> list[dict[str, str]]:
    context = _infer_context(jd_text)
    questions: list[dict[str, str]] = []
    for skill in missing[:3]:
        questions.append(
            {
                "question": f"Explain how you would apply {skill} in a {context} system.",
                "focus_gap": skill,
                "what_good_looks_like": f"Clear tradeoffs, implementation details, and impact using {skill}.",
            }
        )
    return questions


def _build_roadmap_markdown(
    missing: list[str],
    steps: list[dict[str, str]],
    *,
    match_percent: float | None = None,
) -> str:
    if not missing:
        gap_summary = "The resume covers the core requirements, with only minor gaps."
    elif match_percent is not None and match_percent <= 20:
        gap_summary = (
            "The resume currently has low overlap with the JD requirements. "
            f"Priority gaps include {missing[0]}"
            + (f" and {missing[1]}." if len(missing) > 1 else ".")
        )
    elif match_percent is not None and match_percent <= 50:
        gap_summary = (
            "The resume has partial overlap with the JD requirements. "
            f"Priority gaps include {missing[0]}"
            + (f" and {missing[1]}." if len(missing) > 1 else ".")
        )
    else:
        gap_summary = (
            "The resume aligns with the role, but several JD skills are missing. "
            f"Priority gaps include {missing[0]}"
            + (f" and {missing[1]}." if len(missing) > 1 else ".")
        )
    priority = "\n".join([f"- {s}" for s in missing[:3]]) if missing else "- None"
    steps_md = (
        "\n".join(
            [
                f"### Step {i+1}  {step['title']}\n**Why:** {step['why']}\n**Deliverable:** {step['deliverable']}"
                for i, step in enumerate(steps[:3])
            ]
        )
        if steps
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


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.last_usage: dict | None = None

    async def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        if not self.api_key:
            raise ValueError("LLM_API_KEY is required for llm provider")
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "temperature": 0.3 if temperature is None else float(temperature),
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict JSON generator. Output ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        timeout = httpx.Timeout(settings.llm_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                body = ""
                if exc.response is not None:
                    try:
                        body = exc.response.text
                    except Exception:
                        body = ""
                if status == 429:
                    raise ValueError("rate_limited") from exc
                raise ValueError(f"llm_http_{status}: {body}") from exc
            except httpx.RequestError as exc:
                raise ValueError(_format_exception("llm_network_error", exc)) from exc
            data = resp.json()
        self.last_usage = data.get("usage") if isinstance(data, dict) else None
        choices = data.get("choices") if isinstance(data, dict) else None
        if not choices:
            raise ValueError("LLM response missing choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not content:
            raise ValueError("LLM response missing content")
        return content


class LocalLLMProvider:
    name = "local_llm"

    def __init__(self, base_url: str, model: str):
        self.api_key = "local"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.last_usage: dict | None = None

    async def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        url = f"{self.base_url}/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a strict JSON generator. Output ONLY valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3 if temperature is None else float(temperature),
            "stream": False,
            "format": "json",
        }
        timeout = httpx.Timeout(settings.local_llm_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else "unknown"
                body = ""
                if exc.response is not None:
                    try:
                        body = exc.response.text
                    except Exception:
                        body = ""
                raise ValueError(f"local_llm_http_{status}: {body}") from exc
            except httpx.RequestError as exc:
                raise ValueError(_format_exception("local_llm_network_error", exc)) from exc
            data = resp.json()
        message = data.get("message") if isinstance(data, dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not content:
            raise ValueError("Local LLM response missing content")
        self.last_usage = {
            "input_chars": len(prompt),
            "output_chars": len(content),
            "total_tokens": max((len(prompt) + len(content)) // 4, 0),
        }
        return content


class HeuristicProvider:
    name = "heuristic"
    model = "heuristic-v1"
    last_usage: dict | None = None

    async def generate(self, prompt: str, *, temperature: float | None = None) -> str:
        resume_text, jd_text = _extract_payload(prompt)
        (
            missing,
            match_percent,
            match_reason,
            top_priority,
        ) = match_skills(resume_text, jd_text)

        if not missing:
            resume_tokens = _extract_tokens(resume_text)
            jd_tokens = _extract_tokens(jd_text)
            if not jd_tokens:
                match_percent = 0.0
                match_reason = "No skills extracted from JD; match percent not reliable."
            else:
                missing = [t for t in jd_tokens if t not in resume_tokens]
                total = max(len(jd_tokens), 1)
                match_percent = round((1 - (len(missing) / total)) * 100, 2)
                match_reason = (
                    f"Heuristic match on {total} JD skills; {len(missing)} missing identified."
                )
        missing = _rank_missing_skills(missing, jd_text)

        steps = _build_action_steps(missing, jd_text)

        questions = _build_interview_questions(missing, jd_text)

        roadmap = _build_roadmap_markdown(missing, steps, match_percent=match_percent)

        return _to_json(
            {
                "jd_skills_extracted": extract_skills(jd_text),
                "resume_skills_extracted": extract_skills(resume_text),
                "missing_skills": missing,
                "top_priority_skills": top_priority,
                "action_steps": steps,
                "interview_questions": questions,
                "roadmap_markdown": roadmap.strip(),
                "match_percent": match_percent,
                "match_reason": match_reason,
            }
        )


def _extract_payload(prompt: str) -> tuple[str, str]:
    resume_match = re.search(r"RESUME_TEXT:\s*(.*?)\s*JD_TEXT:", prompt, re.S | re.I)
    jd_match = re.search(r"JD_TEXT:\s*(.*)$", prompt, re.S | re.I)
    resume = resume_match.group(1).strip() if resume_match else ""
    jd = jd_match.group(1).strip() if jd_match else ""
    return resume, jd


def _extract_tokens(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9\.\+\#-]{2,}", text.lower())
    allow = {
        "python","java","javascript","typescript","node","nodejs","react","nextjs","vue","angular",
        "fastapi","django","flask","spring","dotnet","c#","c++","go","golang","rust","php","laravel",
        "postgres","postgresql","mysql","sqlite","mongodb","redis","kafka","rabbitmq",
        "docker","kubernetes","k8s","aws","gcp","azure","terraform","ansible",
        "graphql","rest","grpc","sql","nosql","linux","git","ci","cd","devops",
        "microservices","kubernetes","helm","nginx","apache","prometheus","grafana",
        "elasticsearch","opensearch","s3","lambda","iam","oauth","jwt",
    }
    stop = {
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
    }
    tokens = []
    for w in words:
        if w in stop:
            continue
        if w in allow or any(ch in w for ch in ".+#") or any(ch.isdigit() for ch in w):
            tokens.append(w)
    if not tokens:
        tokens = [w for w in words if w not in stop and len(w) >= 3]
    return list(dict.fromkeys(tokens))


def _to_json(payload: dict) -> str:
    import json

    return json.dumps(payload, ensure_ascii=True)


def _build_prompt(
    resume_text: str,
    jd_text: str,
    *,
    deterministic_missing_skills: list[str] | None = None,
    deterministic_top_priority_skills: list[str] | None = None,
    deterministic_match_percent: float | None = None,
    deterministic_match_reason: str | None = None,
) -> str:
    """
    Deterministic-boundary, validator-friendly prompt with structured AI extraction      
    traces, designed to minimize malformed JSON and keep final decisions deterministic.
    """
    prompt = (
        "You are performing a precise Resume Job Description gap analysis.\n\n"

        "Some JDs include company history, others start directly with requirements.\n"
        "Identify technical requirements regardless of position.\n"
        "If the text is short, assume every word may be relevant.\n\n"
        "Return ONLY valid JSON with this exact schema:\n"
        "{"
        "\"jd_skills_extracted\": [\"string\"], "
        "\"resume_skills_extracted\": [\"string\"], "
        "\"missing_skills\": [\"string\"], "
        "\"action_steps\": [{\"title\":\"\",\"why\":\"\",\"deliverable\":\"\"}, ... exactly 3], "
        "\"interview_questions\": [{\"question\":\"\",\"focus_gap\":\"\",\"what_good_looks_like\":\"\"}, ... exactly 3], "
        "\"roadmap_markdown\": \"string\", "
        "\"match_percent\": 0, "
        "\"match_reason\": \"string\""
        "}\n\n"

        "Core rules:\n"
        "- Extract skill candidates from JD and Resume into jd_skills_extracted and resume_skills_extracted.\n"
        "- jd_skills_extracted and resume_skills_extracted are extraction traces only.\n"
        "- missing_skills MUST exactly match DETERMINISTIC_GAP_INPUT.missing_skills.\n"
        "- Keep extraction traces focused on explicit technical skills mentioned in JD/Resume.\n"
        "- match_percent = percentage overlap of JD skills found in Resume (0-100).\n"
        "- match_reason = 1 concise sentence explaining the calculation basis.\n"
        "- action_steps and interview_questions MUST contain exactly 3 items.\n"
        "- In each action_steps[i].why, explain bridge logic from likely existing candidate skill to target missing skill.\n"
        "- Suggested Learning Order MUST follow technical dependencies from foundational to advanced.\n"
        "- Do NOT order by popularity; order by prerequisite chain.\n"
        "- Do NOT include explanations outside JSON.\n\n"
        "- Concrete Steps in roadmap_markdown MUST mirror action_steps (same titles, whys, deliverables).\n\n"

        "roadmap_markdown MUST be valid GitHub-Flavored Markdown with EXACT structure:\n"
        "## Gap Summary\n"
        "- Write 2-3 concise sentences describing readiness gap.\n\n"

        "## Priority Skills to Learn\n"
        "- Provide exactly 3 bullet skill names.\n\n"

        "## Concrete Steps\n"
        "### Step 1 Clear Action Title\n"
        "**Why:** One concise sentence.\n"
        "**Deliverable:** One measurable artifact.\n"
        "(Repeat for Step 2 and Step 3.)\n\n"

        "## Expected Outcomes / Readiness\n"
        "- Provide 2-3 observable readiness bullets.\n\n"

        "## Suggested Learning Order\n"
        "1. Same three priority skills in correct order.\n\n"

        "Hard constraints:\n"
        "- No HTML.\n"
        "- No extra sections.\n"
        "- Deterministic heading order required.\n"
        "- Output MUST be valid JSON only.\n\n"
        "- Use DETERMINISTIC_GAP_INPUT as source of truth for missing_skills, match_percent, and match_reason.\n\n"

        f"RESUME_TEXT:\n{resume_text}\n\n"
        f"JD_TEXT:\n{jd_text}\n\n"
        "DETERMINISTIC_GAP_INPUT:\n"
        + json.dumps(
            {
                "missing_skills": deterministic_missing_skills or [],
                "top_priority_skills": deterministic_top_priority_skills or [],
                "match_percent": (
                    float(deterministic_match_percent)
                    if deterministic_match_percent is not None
                    else 0.0
                ),
                "match_reason": deterministic_match_reason or "",
            },
            ensure_ascii=True,
        )
    )

    # Token-safety truncation
    if len(prompt) > settings.max_prompt_chars:
        logger.warning(
            "prompt_truncated",
            extra={"len": len(prompt), "max": settings.max_prompt_chars},
        )
        prompt = prompt[: settings.max_prompt_chars]

    return prompt


def get_provider() -> LlmProvider:
    provider = settings.llm_provider.lower()
    if provider in {"openai", "compatible", "openai_compatible"}:
        return OpenAICompatibleProvider(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
        )
    return HeuristicProvider()


async def process_gap_analysis(gap_analysis_id) -> None:
    key = str(gap_analysis_id)
    async with _INFLIGHT_GUARD:
        if key in _INFLIGHT_IDS:
            logger.info("gap_analysis_already_inflight", extra={"gap_analysis_id": key})
            return
        _INFLIGHT_IDS.add(key)
    try:
        async with _JOB_SEMAPHORE:
            # Short DB session: fetch input only, then close before LLM network call.
            async with AsyncSessionLocal() as read_session:
                analysis = (
                    await read_session.execute(select(GapAnalysis).where(GapAnalysis.id == gap_analysis_id))
                ).scalars().first()
                if analysis is None:
                    logger.warning(
                        "gap_analysis missing during background processing",
                        extra={"gap_analysis_id": str(gap_analysis_id)},
                    )
                    return
                resume_text = analysis.resume_text
                jd_text = analysis.jd_text

            missing_skills, match_percent, match_reason, top_priority = match_skills(
                resume_text=resume_text,
                jd_text=jd_text,
            )
            prompt = _build_prompt(
                resume_text,
                jd_text,
                deterministic_missing_skills=missing_skills,
                deterministic_top_priority_skills=top_priority,
                deterministic_match_percent=match_percent,
                deterministic_match_reason=match_reason,
            )

        async def _run_with_new_session(
            provider: LlmProvider,
            *,
            fallback_from: str | None = None,
        ) -> None:
            async with AsyncSessionLocal() as write_session:
                await run_gap_analysis_ai(
                    write_session,
                    gap_analysis_id,
                    provider,
                    prompt,
                    fallback_from=fallback_from,
                    raise_on_provider_failure=True,
                )

        provider = get_provider()
        try:
            await _run_with_new_session(provider)
        except ValueError as exc:
            if "rate_limited" in str(exc):
                logger.warning("llm_rate_limited_fallback", extra={"gap_analysis_id": str(gap_analysis_id)})
            elif "local_timeout" in str(exc) or "local_llm_failed" in str(exc):
                logger.warning("local_llm_timeout_fallback", extra={"gap_analysis_id": str(gap_analysis_id)})
                fallback = HeuristicProvider()
                await _run_with_new_session(fallback, fallback_from="local_llm")
                return
            else:
                logger.warning(
                    "llm_error_fallback",
                    extra={"gap_analysis_id": str(gap_analysis_id), "error": sanitize_for_log(exc)},
                )
            try:
                local = LocalLLMProvider(
                    base_url=settings.local_llm_base_url,
                    model=settings.local_llm_model,
                )
                await _run_with_new_session(local, fallback_from="primary")
                return
            except Exception:  # noqa: BLE001
                fallback = HeuristicProvider()
                await _run_with_new_session(fallback, fallback_from="primary_or_local")
                return
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "llm_error_fallback",
                extra={"gap_analysis_id": str(gap_analysis_id), "error": sanitize_for_log(exc)},
            )
            try:
                local = LocalLLMProvider(
                    base_url=settings.local_llm_base_url,
                    model=settings.local_llm_model,
                )
                await _run_with_new_session(local, fallback_from="primary")
                return
            except Exception:  # noqa: BLE001
                fallback = HeuristicProvider()
                await _run_with_new_session(fallback, fallback_from="primary_or_local")
                return
    finally:
        async with _INFLIGHT_GUARD:
            _INFLIGHT_IDS.discard(key)
logger = logging.getLogger("app.llm_service")

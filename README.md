# Gap Architect (Option B)
Full-stack MVP for resume vs JD gap analysis using Next.js + FastAPI + PostgreSQL.

## TL;DR Architecture
This MVP uses AI-assisted extraction with deterministic verification, async non-blocking processing, and fingerprint-based idempotent caching.

- AI performs structured skill extraction from Resume/JD.
- Backend deterministically verifies AI extraction (canonical mapping + text evidence) and computes final diff from verified sets.
- `missing_skills`, `match_percent`, `match_reason` are deterministic from verified skill sets.
- AI generates guidance content (`action_steps`, `interview_questions`, `roadmap_markdown`) under schema validation.
- Caching is idempotent by normalized fingerprint + unique DB constraint.
- Async lifecycle is explicit: `PENDING -> DONE/FAILED_VALIDATION/FAILED_LLM/FAILED_TIMEOUT`.

```text
[SYNC REQUEST PATH]
Resume + JD
   |
POST /gap-analyses
   |
normalize + validate + fingerprint
   |
DB lookup by fingerprint  -------------------------> [DB = source of truth]
   |                                   (gap_analyses / gap_results / llm_runs)
   |-- cache hit (DONE) --> return result (sync)
   |
   |-- cache miss --------> create PENDING in DB (sync)
                            return 201 + analysis_id (sync)
                                      |
                                      v
[ASYNC WORKER PATH]
background worker
   |
AI extraction
   |
deterministic verification + scoring
   |
persist DONE/FAILED_* in DB
```

## How This Meets Assignment Constraints
- **Non-blocking AI processing:** `POST /api/v1/gap-analyses` returns immediately with `PENDING`; analysis continues in background.
- **Validation layer:** AI responses are parsed and schema-validated (Pydantic). Malformed output triggers controlled repair/failure handling instead of frontend crash.
- **Caching requirement:** identical normalized Resume+JD is deduplicated by fingerprint (DB unique constraint) and served from existing result.
- **Edge-case handling:** malformed JSON, provider failure, timeout, and duplicate concurrent submissions are handled explicitly.

## Real Failure Cases Found During Development
1. **LLM returned JSON wrapped in markdown fences** (```json ... ```).  
   Fix: robust parser extracts first JSON object before schema validation.
2. **Duplicate concurrent submissions** could race on the same input.  
   Fix: unique fingerprint + atomic status transition (CAS) + idempotent retry behavior.
3. **Noisy/long JD caused unstable inferred gaps.**  
   Fix: deterministic verification (taxonomy + text evidence) before final scoring fields are persisted.

## Key Engineering Decisions
- Deterministic final scoring prevents LLM numeric drift and keeps results auditable.
- Async background flow satisfies non-blocking UX and assignment bottleneck constraint.
- Fingerprint cache in DB provides simple, reliable idempotency without extra infra complexity.
- AI is used for extraction + guidance generation, while final decision fields remain verification-driven.
- This architecture ensures LLM uncertainty cannot directly affect final scoring decisions.

## Why This Design
- Stable scoring: decision fields are verification-driven, not raw LLM guesses.
- Reliability: explicit async lifecycle + timeout/failure states avoid ambiguous processing.
- Auditability: each run is persisted with status and metadata in PostgreSQL.

## End-to-End Flow
1. Submit Resume + JD to `POST /api/v1/gap-analyses`.
2. Normalize input, validate quality, compute fingerprint, check cache.
3. If cache hit `DONE`: return result immediately.
4. If miss: create `PENDING`, enqueue background processing.
5. Worker runs AI extraction -> deterministic verification/normalization -> final diff/scoring -> constrained generation.
6. Persist result and expose status via `GET /api/v1/gap-analyses/{id}`.

## Production Boundary
This submission focuses on production-aware MVP reliability:
deterministic scoring, idempotent caching, async lifecycle, and timeout recovery.

Trade-off: advanced durability layers (distributed queues, shared cache, full metrics/alerting) are not included in this MVP to keep implementation lean and verifiable.
Mechanisms such as fallback chain, timeout sweep, and taxonomy normalization are intentionally lightweight and kept within MVP boundary.

## Domain Scope
This MVP targets software engineering and technical roles,
where deterministic skill normalization and scoring are reliable.

Extending coverage to additional domains is supported
through taxonomy expansion without changing the scoring pipeline.

## Edge Cases Handled
- Malformed AI output: strict parse + schema validation + repair attempt.
- Provider issues: fallback (`primary -> local_llm -> heuristic`).
- Long/noisy JD: adaptive clipping.
- Duplicate concurrent requests: race-safe fingerprint flow and in-flight guard.
- Stuck jobs: TTL-based transition to `FAILED_TIMEOUT`.

## Data & Async Structure
- Core tables:
  - `gap_analyses`: fingerprint, status, error.
  - `gap_results`: final missing skills, score/reason, roadmap outputs.
  - `llm_runs`: provider/model/request/response/status/duration.
  - `jd_clean_runs`: cleaning strategy audit.
- Async behavior:
  - API returns non-blocking.
  - Worker concurrency capped by `MAX_CONCURRENT_GAP_JOBS`.
  - Stuck TTL configured by `PENDING_TIMEOUT_SECONDS`.
  - Retry backoff configured by `RETRY_COOLDOWN_SECONDS`.

## Concurrency & Test Guarantees
The system enforces idempotent and race-safe processing for identical submissions:

- A unique fingerprint constraint ensures only one analysis row per normalized Resume+JD.
- Atomic status transitions prevent duplicate background execution.
- Parallel identical requests must produce:
  - exactly one persisted `gap_result`
  - exactly one successful `llm_run`
  - all other requests returning the same cached analysis id or `PENDING`.

Concurrency correctness is validated at the database level.

SQLite in-memory is not suitable for this verification because:
- in-memory databases are connection-scoped and not shared,
- transactional and locking semantics differ from PostgreSQL.

Therefore, concurrency invariants are verified against PostgreSQL,
which is the target production database and source of transactional truth.

## Status State Machine
| Status | What it means | Transition |
|--------|---------------|------------|
| `PENDING` | Job queued or processing in background | `DONE`, `FAILED_VALIDATION`, `FAILED_LLM`, `FAILED_TIMEOUT` |
| `DONE` | Analysis complete and persisted | n/a |
| `FAILED_VALIDATION` | AI output failed contract validation | retry -> `PENDING` |
| `FAILED_LLM` | Provider call or processing failure | retry -> `PENDING` |
| `FAILED_TIMEOUT` | Processing exceeded timeout threshold | retry -> `PENDING` |

## AI Output Validation
- Pydantic schema gate for AI result contract.
- Final decision fields are computed deterministically from verified extraction:
  - `missing_skills`
  - `top_priority_skills`
  - `match_percent`
  - `match_reason`
- AI extraction is accepted only when it passes deterministic verification against taxonomy and source text evidence.

## AI Tooling Workflow
- Used Cursor, Windsurf, GPT, and Claude for:
  - architecture discussion and trade-off checks
  - code review on race/consistency/parser failures
  - refactoring and boilerplate acceleration
- Final behavior was verified by tests + runtime logs (not prompt assumptions).

## Testing Strategy
- Tooling: `pytest` + `httpx` async client + SQLAlchemy async sessions.
- Unit tests: parser/validation paths, deterministic scoring behavior, and failure handling.
- Concurrency tests: parallel submissions validating idempotent fingerprint flow and CAS transitions.
- Contract tests: API status lifecycle and response schema consistency.
- Manual E2E: backend logs + DB inspection (`gap_analyses`, `gap_results`, `llm_runs`).

## Known Limitations
- Taxonomy coverage is currently tuned for software/technical roles; non-technical domains need taxonomy expansion.
- Heuristic fallback is intentionally simple and may be less domain-aware than primary provider outputs.
- Background processing uses in-process workers (no distributed queue), so horizontal resilience is limited by deployment model.

## Video Walkthrough Checklist
- Show non-blocking submit (`PENDING` immediately).
- Show low-match and high-match runs.
- Show repeated identical input returns cached result.
- Briefly explain: edge-case handling, DB/async design, AI output validation.

## Local Setup (Docker)
1. `cp .env.dev.example .env`
2. `docker compose up -d --build`
3. Optional local Ollama: `docker compose --profile local-llm up -d --build`

Endpoints:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Health: `http://localhost:8000/health`

### LLM Provider Setup
Configure your LLM provider in .env (HuggingFace is the default).

Optional runtime paths:
- Local LLM fallback (LOCAL_LLM_*) for offline/local reliability.
- Heuristic fallback when external provider credentials are missing or invalid.

## Production Compose
1. `cp .env.prod.example .env.production`
2. `docker compose --env-file .env.production -f docker-compose.prod.yml up -d --build`

## App-Level Setup Docs
- Backend: `backend/README.md`
- Frontend: `frontend/README.md`

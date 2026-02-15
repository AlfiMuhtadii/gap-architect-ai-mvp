# Backend Setup

## Prerequisites
- Python 3.11+
- PostgreSQL (local) or Docker Postgres

## 1. Environment
- Copy template:
  - `cp .env.dev.example .env`
- Minimal required values:
  - `POSTGRES_HOST`
  - `POSTGRES_PORT`
  - `POSTGRES_DB`
  - `POSTGRES_USER`
  - `POSTGRES_PASSWORD`
  - or set `DATABASE_URL` and `ASYNC_DATABASE_URL` directly
- Optional LLM:
  - `LLM_PROVIDER=compatible`
  - `LLM_BASE_URL=https://router.huggingface.co`
  - `LLM_API_KEY=...`
  - `LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct`

## 2. Install Dependencies
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run Migrations
```bash
alembic upgrade head
```

## 4. Run API
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## 5. Verify
- Health check: `http://localhost:8000/health`
- API base: `http://localhost:8000/api/v1`

## Test
```bash
.venv\Scripts\python -m pytest -q
```

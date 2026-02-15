from pydantic import BaseModel
import os
from urllib.parse import quote_plus
from pathlib import Path
from dotenv import load_dotenv


def _build_database_url(prefix: str, host: str, port: str, db: str, user: str, password: str) -> str:
    safe_password = quote_plus(password)
    return f"{prefix}://{user}:{safe_password}@{host}:{port}/{db}"


def _getenv_nonempty(key: str) -> str | None:
    value = os.getenv(key)
    if value is None:
        return None
    if not str(value).strip():
        return None
    return value


_base_dir = Path(__file__).resolve().parents[2]
load_dotenv(_base_dir / ".env")
load_dotenv(_base_dir.parent / ".env")


class Settings(BaseModel):
    app_name: str = os.getenv("APP_NAME", "gap-architect")
    api_v1_str: str = os.getenv("API_V1_STR", "/api/v1")

    postgres_host: str = _getenv_nonempty("POSTGRES_HOST") or "localhost"
    postgres_port: str = _getenv_nonempty("POSTGRES_PORT") or "5432"
    postgres_db: str = _getenv_nonempty("POSTGRES_DB") or "gap_architect"
    postgres_user: str = _getenv_nonempty("POSTGRES_USER") or "postgres"
    postgres_password: str = _getenv_nonempty("POSTGRES_PASSWORD") or ""

    database_url: str = _getenv_nonempty("DATABASE_URL") or _build_database_url(
        "postgresql+psycopg",
        postgres_host,
        postgres_port,
        postgres_db,
        postgres_user,
        postgres_password,
    )
    async_database_url: str = _getenv_nonempty("ASYNC_DATABASE_URL") or _build_database_url(
        "postgresql+asyncpg",
        postgres_host,
        postgres_port,
        postgres_db,
        postgres_user,
        postgres_password,
    )
    cors_origins: list[str] = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
        if origin.strip()
    ]

    max_resume_chars: int = int(os.getenv("MAX_RESUME_CHARS", "20000"))
    max_jd_chars: int = int(os.getenv("MAX_JD_CHARS", "20000"))
    llm_timeout_seconds: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    local_llm_timeout_seconds: float = float(os.getenv("LOCAL_LLM_TIMEOUT_SECONDS", "120"))
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "120"))
    rate_limit_max_keys: int = int(os.getenv("RATE_LIMIT_MAX_KEYS", "10000"))
    rate_limit_key_header: str = os.getenv("RATE_LIMIT_KEY_HEADER", "")
    redis_url: str = os.getenv("REDIS_URL", "")
    llm_provider: str = os.getenv("LLM_PROVIDER", "compatible")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")
    llm_base_url: str = os.getenv("LLM_BASE_URL", "https://router.huggingface.co")
    llm_model: str = os.getenv("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
    local_llm_base_url: str = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
    local_llm_model: str = os.getenv("LOCAL_LLM_MODEL", "llama3.1")
    max_concurrent_gap_jobs: int = int(os.getenv("MAX_CONCURRENT_GAP_JOBS", "4"))
    pending_timeout_seconds: int = int(os.getenv("PENDING_TIMEOUT_SECONDS", "600"))
    retry_cooldown_seconds: int = int(os.getenv("RETRY_COOLDOWN_SECONDS", "15"))
    canonical_skills_path: str = os.getenv(
        "CANONICAL_SKILLS_PATH",
        str(_base_dir / "datasets" / "canonical_skills.txt"),
    )
    max_prompt_chars: int = int(os.getenv("MAX_PROMPT_CHARS", "60000"))


settings = Settings()

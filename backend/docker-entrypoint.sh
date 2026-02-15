#!/bin/sh
set -eu

echo "[backend] waiting for database..."
python - <<'PY'
import os
import time
from sqlalchemy import create_engine, text

database_url = os.getenv("DATABASE_URL", "").strip()
if not database_url:
    raise SystemExit("DATABASE_URL is required")

last_error = None
for _ in range(60):
    try:
        with create_engine(database_url, pool_pre_ping=True).connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[backend] database is ready")
        break
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        time.sleep(1)
else:
    raise SystemExit(f"database not ready after 60s: {last_error}")
PY

echo "[backend] running alembic migrations..."
alembic upgrade head

echo "[backend] starting app..."
exec "$@"

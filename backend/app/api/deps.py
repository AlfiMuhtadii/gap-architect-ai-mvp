from collections.abc import AsyncGenerator
from app.db.session import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.core.config import settings


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        if settings.async_database_url.startswith("postgresql"):
            try:
                await session.execute(text("SET LOCAL statement_timeout = 5000"))
            except Exception:
                pass
        yield session

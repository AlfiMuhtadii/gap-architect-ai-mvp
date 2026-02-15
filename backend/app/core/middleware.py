import time
import asyncio
import uuid
from collections import defaultdict, deque
from typing import Deque, DefaultDict

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from app.core.config import settings
import logging
from typing import Optional

try:
    from redis.asyncio import Redis  # type: ignore
except Exception:  # noqa: BLE001
    Redis = None  # type: ignore


logger = logging.getLogger("app.middleware")


class RequestIdMiddleware(BaseHTTPMiddleware):
    header_name = "X-Request-ID"

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(self.header_name, str(uuid.uuid4()))
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers[self.header_name] = request_id
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.window_seconds = 60
        self.max_requests = settings.rate_limit_per_minute
        self._requests: DefaultDict[str, Deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()
        self._redis: Optional["Redis"] = None
        self._cleanup_counter = 0
        if settings.redis_url and Redis is not None:
            self._redis = Redis.from_url(settings.redis_url, decode_responses=True)

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if self.max_requests <= 0:
            return await call_next(request)

        header_key = settings.rate_limit_key_header
        header_val = request.headers.get(header_key) if header_key else None
        ip = request.client.host if request.client else "unknown"
        key = f"{ip}:{header_val}" if header_val else ip
        now = time.monotonic()

        if self._redis is not None:
            try:
                now_ms = int(time.time() * 1000)
                window_ms = self.window_seconds * 1000
                redis_key = f"rate:{key}"
                lua = """
                local key = KEYS[1]
                local now = tonumber(ARGV[1])
                local window = tonumber(ARGV[2])
                local limit = tonumber(ARGV[3])
                redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
                local count = redis.call('ZCARD', key)
                if count >= limit then
                    return 0
                end
                redis.call('ZADD', key, now, now)
                redis.call('EXPIRE', key, math.ceil(window / 1000))
                return 1
                """
                allowed = await self._redis.eval(lua, 1, redis_key, now_ms, window_ms, self.max_requests)
                if int(allowed) == 0:
                    logger.warning("rate_limited", extra={"request_id": getattr(request.state, "request_id", None)})
                    return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)
            except Exception:  # noqa: BLE001
                logger.warning("rate_limit_redis_error", extra={"request_id": getattr(request.state, "request_id", None)})
                # fall back to in-memory
        # in-memory fallback
        async with self._lock:
            q = self._requests[key]
            while q and now - q[0] > self.window_seconds:
                q.popleft()

            if len(q) >= self.max_requests:
                logger.warning("rate_limited", extra={"request_id": getattr(request.state, "request_id", None)})
                return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429)

            q.append(now)
            self._cleanup_counter += 1
            if self._cleanup_counter % 500 == 0:
                self._cleanup_stale(now)
        return await call_next(request)

    def _cleanup_stale(self, now: float) -> None:
        window = self.window_seconds
        # remove stale keys
        stale_keys = [k for k, q in self._requests.items() if not q or now - q[-1] > window]
        for k in stale_keys:
            self._requests.pop(k, None)
        # enforce max keys
        max_keys = settings.rate_limit_max_keys
        if max_keys > 0 and len(self._requests) > max_keys:
            items = sorted(
                ((q[-1], k) for k, q in self._requests.items() if q),
                key=lambda x: x[0],
            )
            for _, k in items[: len(self._requests) - max_keys]:
                self._requests.pop(k, None)

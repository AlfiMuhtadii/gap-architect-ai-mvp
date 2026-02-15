from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import configure_logging
from app.core.middleware import RequestIdMiddleware, RateLimitMiddleware
from app.api.routes import router as api_router

configure_logging()

app = FastAPI(title=settings.app_name)
app.add_middleware(RequestIdMiddleware)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok", "app": settings.app_name}



app.include_router(api_router, prefix=settings.api_v1_str)

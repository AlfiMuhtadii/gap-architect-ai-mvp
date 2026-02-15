from fastapi import APIRouter
from app.api.routes.gap_analyses import router as gap_analyses_router

router = APIRouter()
router.include_router(gap_analyses_router)

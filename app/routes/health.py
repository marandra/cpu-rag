"""Health check endpoint."""

from fastapi import APIRouter

from app.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    from app.main import app_state

    return HealthResponse(
        status="healthy",
        model=app_state.model_name,
        procedures=sorted(app_state.procedures),
    )

from fastapi import APIRouter, Request

from recommender_system.api.models import HealthResponse

router = APIRouter(
    prefix="",
    tags=["health"],
)


@router.get("/")
def home():
    """
    Root endpoint - confirms the API is running.
    """
    return {
        "message": "Anime Recommendation API is running!",
        "docs": "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthResponse)
def health_check(request: Request):
    """
    Health check endpoint for monitoring and Kubernetes probes.
    
    Returns the current status of the API and pipeline readiness.
    """
    pipeline_ready = (
        hasattr(request.app.state, "pipeline") 
        and request.app.state.pipeline is not None
    )
    
    return HealthResponse(
        status="healthy" if pipeline_ready else "degraded",
        pipeline_ready=pipeline_ready,
    )

from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from recommender_system.api.models import (
    RecommendationRequest,
    RecommendationResponse,
)
from recommender_system.utils.custom_exception import CustomException
from recommender_system.utils.logger import get_logger

logger = get_logger(__name__)

# Rate limiter for this router
limiter = Limiter(key_func=get_remote_address)

router = APIRouter(
    prefix="/api/v1",
    tags=["recommendations"],
)


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    responses={
        200: {
            "description": "Successful recommendation",
            "content": {
                "application/json": {
                    "example": {
                        "answer": "(Full AI-generated recommendation will appear here - this is just an example format)",
                        "query": "action anime with strong protagonist"
                    }
                }
            }
        },
        400: {"description": "Bad request - invalid query"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"},
        503: {"description": "Service unavailable - pipeline not ready"},
    }
)
@limiter.limit("10/minute")
def recommend(request: Request, body: RecommendationRequest):
    """
    Get anime recommendations based on your query.
    
    This endpoint uses RAG (Retrieval Augmented Generation) to:
    1. Search the anime database for relevant titles
    2. Generate personalized recommendations using AI
    
    **Rate Limit:** 10 requests per minute per IP address.
    """
    # Check if pipeline is ready
    if not hasattr(request.app.state, "pipeline") or request.app.state.pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Recommendation service is not ready. Please try again later."
        )
    
    try:
        pipeline = request.app.state.pipeline
        result = pipeline.recommend(body.query)
        
        return RecommendationResponse(
            answer=result,
            query=body.query,
        )

    except CustomException as e:
        logger.warning(f"Custom exception for query '{body.query}': {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error for query '{body.query}': {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error. Please try again later."
        )

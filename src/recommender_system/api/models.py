from pydantic import BaseModel, Field, field_validator


class RecommendationRequest(BaseModel):
    """Request model for anime recommendations."""
    query: str = Field(
        ...,
        min_length=3,
        max_length=500,
        description="Anime search query (e.g., 'action anime with strong protagonist')"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Strip whitespace from query."""
        return v.strip()


class RecommendationResponse(BaseModel):
    """Response model for anime recommendations."""
    answer: str = Field(..., description="AI-generated anime recommendations")
    query: str = Field(..., description="Original user query (echoed back)")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    status: str = Field(..., description="Service health status")
    pipeline_ready: bool = Field(..., description="Whether the recommendation pipeline is loaded")


class ErrorResponse(BaseModel):
    """Standard error response model."""
    detail: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error that occurred")

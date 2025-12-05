from pydantic import BaseModel

class RecommendationRequest(BaseModel):
    query: str

class RecommendationResponse(BaseModel):
    answer: str

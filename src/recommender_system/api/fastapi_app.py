from fastapi import FastAPI, HTTPException
from recommender_system.pipeline.recommend_pipeline import AnimeRecommendationPipeline
from recommender_system.api.models import RecommendationRequest, RecommendationResponse
from recommender_system.utils.custom_exception import CustomException

app = FastAPI(
    title="Anime Recommender API",
    description="RAG-powered Anime Recommendation System using Groq + ChromaDB",
    version="1.0.0"
)

# Initialize pipeline once (FAST!!)
pipeline = AnimeRecommendationPipeline()

# human readable  
@app.get("/")
def home():
    return {"message": "Anime Recommendation API is running!"}

# machine readable
@app.get('/health')
def health_check():
    return {
        'status': 'OK',
        'pipeline_loaded': pipeline is not None
    }

@app.post("/recommend", response_model=RecommendationResponse)
def recommend(request: RecommendationRequest):
    try:
        result = pipeline.recommend(request.query)
        return RecommendationResponse(answer=result)

    except CustomException as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")



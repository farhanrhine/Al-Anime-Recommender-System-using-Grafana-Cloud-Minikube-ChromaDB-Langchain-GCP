import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from recommender_system.pipeline.recommend_pipeline import AnimeRecommendationPipeline
from recommender_system.api.routes import health_router, recommendations_router
from recommender_system.utils.logger import get_logger

logger = get_logger(__name__)

# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Modern FastAPI lifespan event handler.
    Handles startup and shutdown logic cleanly.
    """
    # Startup
    logger.info("Starting up: Loading recommendation pipeline...")
    try:
        app.state.pipeline = AnimeRecommendationPipeline()
        logger.info("Pipeline loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load pipeline: {e}")
        app.state.pipeline = None
    
    yield  # App is running
    
    # Shutdown
    logger.info("Shutting down Anime Recommender API...")


# OpenAPI tags for better documentation
tags_metadata = [
    {
        "name": "health",
        "description": "Health check and monitoring endpoints.",
    },
    {
        "name": "recommendations",
        "description": "Get AI-powered anime recommendations based on your preferences.",
    },
]

app = FastAPI(
    title="Anime Recommender API",
    description="""
## RAG-powered Anime Recommendation System

This API uses **Groq LLM + ChromaDB** to provide intelligent anime recommendations 
based on your preferences and queries.

### Features
- Semantic search through anime database
- AI-generated personalized recommendations
- Fast response times with cached embeddings

### Example Query
> "I want an action anime with a strong protagonist and good fight scenes"

### API Versioning
- Current version: `v1`
- Base path: `/api/v1`
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata,
    lifespan=lifespan,
)

# Add rate limiter to app state and exception handler
app.state.limiter = limiter
app.add_exception_handler(
    RateLimitExceeded, 
    lambda request, exc: {"detail": "Rate limit exceeded. Please try again later."}
)
app.add_middleware(SlowAPIMiddleware)

# CORS Middleware - adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics for Grafana monitoring
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    Middleware to log all incoming requests with timing information.
    """
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Skip logging for metrics endpoint to avoid noise
    if request.url.path != "/metrics":
        logger.info(
            f"{request.method} {request.url.path} - "
            f"Status: {response.status_code} - "
            f"Duration: {process_time:.3f}s"
        )
    
    # Add timing header
    response.headers["X-Process-Time"] = f"{process_time:.3f}"
    return response


# ============================================
# REGISTER ROUTERS
# ============================================

app.include_router(health_router)
app.include_router(recommendations_router)

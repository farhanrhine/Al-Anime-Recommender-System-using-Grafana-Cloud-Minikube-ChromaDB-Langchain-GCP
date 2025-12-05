# Use Python 3.12 base image + UV (Debian Bookworm slim)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy project files

# Copy only dependency files first for Docker caching
COPY pyproject.toml uv.lock ./

# Now copy the application code (lightweight)
COPY src ./src

# Install dependencies using UV (super fast)
RUN uv sync --frozen

# Expose the application port 8000 for FastAPI , for differnt needs can be changed
EXPOSE 8000

# Run FastAPI using uvicorn (no reload in Docker)
CMD ["uv", "run", "uvicorn", "recommender_system.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]






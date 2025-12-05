"""
🎌 AI Anime Recommender - Streamlit Frontend
"""

import streamlit as st
import requests

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="AI Anime Recommender",
    page_icon="🎌",
    layout="centered",
)

# =============================================================================
# CONFIG
# =============================================================================

# Define the FastAPI endpoint
API_BASE_URL = "http://localhost:8000" # Use this for local testing
# API_BASE_URL = "http://34.226.152.222:8000"  # Use this for deployed API

# =============================================================================
# API FUNCTIONS
# =============================================================================
def check_api_health() -> bool:
    """Check if the FastAPI backend is healthy."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        # Just check if API responds with 200
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def get_recommendation(query: str) -> dict:
    """Get anime recommendation from the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend",  # Changed from /api/v1/recommend
            json={"query": query},
            timeout=60,
        )
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        elif response.status_code == 429:
            return {"success": False, "error": "Rate limit exceeded. Wait a moment."}
        else:
            return {"success": False, "error": response.json().get("detail", "Error")}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out."}
    except Exception as e:
        return {"success": False, "error": f"Connection error: {e}"}


# =============================================================================
# MAIN APP
# =============================================================================

# Header
st.title("🎌 AI Anime Recommender")
st.caption("Get personalized anime recommendations powered by AI")

# API Status
if check_api_health():
    st.success("✅ API is online", icon="🟢")
else:
    st.error("❌ API is offline. Start the FastAPI server first.")
    st.code("uv run uvicorn recommender_system.api.fastapi_app:app --reload")
    st.stop()

st.divider()

# Query Input
query = st.text_area(
    "What kind of anime are you looking for?",
    placeholder="Example: Action anime with a strong protagonist and epic fights...",
    height=100,
)

# Submit Button
if st.button("🔮 Get Recommendations", type="primary", use_container_width=True):
    if not query or len(query.strip()) < 3:
        st.warning("Please enter at least 3 characters.")
    else:
        with st.spinner("Finding recommendations..."):
            result = get_recommendation(query.strip())
        
        if result["success"]:
            st.divider()
            st.subheader("✨ Recommendations")
            st.markdown(result["data"]["answer"])
        else:
            st.error(result["error"])

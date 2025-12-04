import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Models
MODEL_NAME = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Validate
if GROQ_API_KEY is None:
    raise ValueError("❌ Missing GROQ_API_KEY in .env")
if HF_TOKEN is None:
    raise ValueError("❌ Missing HUGGINGFACEHUB_API_TOKEN in .env")

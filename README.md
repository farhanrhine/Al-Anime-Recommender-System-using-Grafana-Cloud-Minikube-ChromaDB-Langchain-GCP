# 🎌 AI Anime Recommender System

RAG-powered Anime Recommendation System using **LangChain**, **ChromaDB**, **Groq LLM**, deployed on **GKE Autopilot** with **LangSmith** tracing.

![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/🦜_LangChain-RAG-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-orange)
![Groq](https://img.shields.io/badge/Groq-Llama_3.1-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.123-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![GCP](https://img.shields.io/badge/Google_Cloud-GCP-4285F4?logo=googlecloud&logoColor=white)
![Kubernetes](https://img.shields.io/badge/Kubernetes-GKE-326CE5?logo=kubernetes&logoColor=white)

![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-blue)

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Groq Llama 3.1 8B |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector Store | ChromaDB |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Package Manager | UV |
| Container | Docker |
| Deployment | GKE Autopilot (via Cloud Shell) |
| Observability | LangSmith |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Streamlit     │────▶│    FastAPI      │────▶│   RAG Pipeline  │
│   Frontend      │     │    Backend      │     │                 │
└─────────────────┘     └─────────────────┘     │  ChromaDB       │
                                                │       ▼         │
                                                │  HuggingFace    │
                                                │  Embeddings     │
                                                │       ▼         │
                                                │  Groq LLM       │
                                                │  (Llama 3.1)    │
                                                └────────┬────────┘
                                                         │
                        ┌────────────────────────────────┴────────────────────────────────┐
                        ▼                                                                 ▼
                ┌──────────────┐                                                 ┌──────────────┐
                │  LangSmith   │                                                 │     GKE      │
                │  Tracing     │                                                 │  Autopilot   │
                └──────────────┘                                                 └──────────────┘
```

---

## 📦 Project Structure

```
ai_anime_recommender_system/
├── src/recommender_system/
│   ├── api/
│   │   ├── fastapi_app.py      # FastAPI endpoints
│   │   └── models.py           # Pydantic models
│   ├── pipeline/
│   │   ├── build_embedding_pipeline.py   # Build vector DB
│   │   └── recommend_pipeline.py         # RAG pipeline
│   ├── config/settings.py      # API keys & model config
│   ├── vector_store.py         # ChromaDB operations
│   ├── recommender.py          # LangChain RAG chain
│   └── prompt_template.py      # LLM prompts
├── app/streamlit_app.py        # Streamlit UI
├── docker/Dockerfile           # Docker build
├── kubernetes/                 # K8s manifests (GKE)
├── data/                       # Anime dataset
└── chroma_db/                  # Vector store (generated)
```

---

## 🚀 Local Development

### 1. Setup Environment

```bash
uv sync
uv pip install -e .
```

### 2. Create `.env`

```env
GROQ_API_KEY=your_groq_key
HUGGINGFACEHUB_API_TOKEN=your_hf_token

# Optional: LangSmith
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key
LANGCHAIN_PROJECT=anime-recommender
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

### 3. Activate Virtual Environment

```powershell
.venv/Scripts/activate
```

### 4. Build Vector Database (One-time)

```bash
uv run src/recommender_system/pipeline/build_embedding_pipeline.py
```

### 5. Run FastAPI Server

```bash
uvicorn recommender_system.api.fastapi_app:app --reload
```

### 6. Run Streamlit UI (Separate Terminal)

```bash
uv run streamlit run app/streamlit_app.py
```

---

## 🐳 Docker

### 📥 Pull from DockerHub

```bash
docker pull farhanrhine/anime-recommender-api:latest
```

[![Docker Hub](https://img.shields.io/docker/pulls/farhanrhine/anime-recommender-api?style=flat&logo=docker&label=Docker%20Pulls)](https://hub.docker.com/r/farhanrhine/anime-recommender-api)

### Build Image (Optional)

```bash
docker build -t farhanrhine/anime-recommender-api -f docker/Dockerfile .
```

### Run Container

> **Note:** API keys are passed via `.env` file at runtime, not baked into the image. ChromaDB is mounted as a volume.

```bash
docker run -p 8000:8000 --env-file .env -v ${PWD}/chroma_db:/app/chroma_db farhanrhine/anime-recommender-api
```

### Push to DockerHub

```bash
docker login
docker push farhanrhine/anime-recommender-api:latest
```

---

## ☸️ GKE Deployment

### Kubernetes Manifests

| File | Purpose |
|------|---------|
| `storageclass.yaml` | GCE persistent disk storage class |
| `pv.yaml` | Binds to `ai-anime-chroma-disk` |
| `pvc.yaml` | Claims 20GB for ChromaDB |
| `deployment.yaml` | FastAPI pod with secrets & volume |
| `service.yaml` | LoadBalancer (external IP) |

### Create Secrets

```bash
kubectl create secret generic anime-secrets \
  --from-literal=GROQ_API_KEY=xxx \
  --from-literal=HUGGINGFACEHUB_API_TOKEN=xxx \
  --from-literal=LANGCHAIN_API_KEY=xxx \
  --from-literal=LANGCHAIN_TRACING_V2=true \
  --from-literal=LANGCHAIN_ENDPOINT=https://api.smith.langchain.com \
  --from-literal=LANGCHAIN_PROJECT=anime-recommender
```

### Deploy

```bash
kubectl apply -f kubernetes/
```

### Get External IP

```bash
kubectl get svc anime-service
```

---

## ☁️ Manual Deployment (GCP Cloud Shell)

This project is deployed manually using GCP Cloud Shell:

### 1. Clone Repository in Cloud Shell

```bash
git clone https://github.com/farhanrhine/Al-Anime-Recommender-using-Grafana-Cloud-Minikube-ChromaDB-Langchain-GCP.git
cd Al-Anime-Recommender-using-Grafana-Cloud-Minikube-ChromaDB-Langchain-GCP
```

### 2. Connect to GKE Cluster

```bash
gcloud container clusters get-credentials <YOUR_CLUSTER_NAME> --region <YOUR_REGION> --project <YOUR_PROJECT_ID>
```

### 3. Create Secrets & Deploy

```bash
kubectl create secret generic anime-secrets \
  --from-literal=GROQ_API_KEY=xxx \
  --from-literal=HUGGINGFACEHUB_API_TOKEN=xxx \
  --from-literal=LANGCHAIN_API_KEY=xxx \
  --from-literal=LANGCHAIN_TRACING_V2=true \
  --from-literal=LANGCHAIN_ENDPOINT=https://api.smith.langchain.com \
  --from-literal=LANGCHAIN_PROJECT=anime-recommender

kubectl apply -f kubernetes/
```

### 4. Get External IP

```bash
kubectl get svc anime-service
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health message |
| `/health` | GET | Pipeline status |
| `/recommend` | POST | Get recommendations |

### Example Request

```bash
# Local
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Action anime with epic fights"}'

# GCP Deployment
curl -X POST http://136.111.237.172:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Action anime with epic fights"}'
```

---

## 👨‍💻 Author

**Farhan**

- GitHub: [@farhanrhine](https://github.com/farhanrhine)
- Email: mohammadfarhanalam09@gmail.com

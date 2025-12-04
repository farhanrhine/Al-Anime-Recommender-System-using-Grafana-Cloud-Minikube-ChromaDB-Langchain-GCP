
---

# 📦 **Project Structure — Explained**

```
al_anime_recommender_system/
│
├── pyproject.toml
├── uv.lock
├── .env
├── .gitignore
├── README.md
│
├── data/
│   └── anime_with_synopsis.csv
│
├── src/
│   └── al_anime_recommender_system/
│       ├── __init__.py
│
│       ├── data_loader.py
│       ├── vector_store.py
│       ├── recommender.py
│       ├── prompt_template.py
│
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── logger.py
│       │   └── custom_exception.py
│
│       ├── config/
│       │   ├── __init__.py
│       │   └── settings.py
│
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── build_embedding_pipeline.py
│       │   └── recommend_pipeline.py
│
├── app/
│   └── streamlit_app.py
│
├── docker/
│   ├── Dockerfile
│   └── requirements.txt
│
└── kubernetes/
    ├── deployment.yaml
    ├── service.yaml
    └── monitoring.yaml
```

---

# 🧩 **Folder & File Purpose (Detailed Documentation)**

## 📌 **Root Level Files**

### **`pyproject.toml`**

* UV (package manager) configuration.
* Stores project metadata & dependencies.
* Replaces `requirements.txt` for local development.

### **`uv.lock`**

* Auto-generated lockfile for exact dependency versions.
* Ensures reproducible builds.

### **`.env`**

* Stores secrets such as:

  * `GROQ_API_KEY`
  * `HUGGINGFACEHUB_API_TOKEN`
* Never push this to GitHub.

### **`.gitignore`**

* Excludes sensitive or unnecessary files.
* Includes `.env`, `.venv/`, `__pycache__/`, etc.

---

# 📁 **`data/`**

### **`anime_with_synopsis.csv`**

* Raw dataset containing anime titles + descriptions.
* Used for embedding generation and vector storage.

---

# 📁 **`src/al_anime_recommender_system/` (Main Logic)**

### **`__init__.py`**

* Makes the directory a Python package.
* Needed so you can import modules cleanly.

---

## 🔹 **Core Modules**

### **`data_loader.py`**

Loads and preprocesses dataset:

* Reads CSV
* Cleans text
* Prepares anime descriptions
* Returns Pandas DataFrame

### **`vector_store.py`**

Handles:

* Creating ChromaDB client
* Storing embeddings
* Retrieving vectors
* Searching similar anime

### **`recommender.py`**

Main recommendation logic:

* Embedding generation
* Similarity search
* Ranking and returning top recommendations

### **`prompt_template.py`**

Stores LLM prompts for:

* Explanation generation
* Personalized anime recommendations

---

## 🔹 **`utils/` — Helper Functions**

### **`logger.py`**

* Configures logging
* Writes logs with timestamps
* Helps with debugging in ML pipelines

### **`custom_exception.py`**

* Custom error classes
* Cleaner error handling across project

---

## 🔹 **`config/` — App Settings**

### **`settings.py`**

Loads secrets and configuration:

* `.env` reading using `dotenv`
* API keys
* Global constants

---

## 🔹 **`pipeline/` — ML Pipelines**

### **`build_embedding_pipeline.py`**

Pipeline for:

* Loading data
* Generating embeddings using SentenceTransformers
* Storing vectors into ChromaDB

### **`recommend_pipeline.py`**

Pipeline for end-to-end recommendation:

* Takes user input
* Gets similar anime
* Uses Groq Llama3 to generate reasoning/summary

---

# 🎨 **`app/` — Streamlit Frontend**

### **`streamlit_app.py`**

* User interface for entering anime titles
* Calls backend recommender
* Displays recommendations + descriptions
* Clean UI with Streamlit widgets

---

# 🐳 **`docker/` — Containerization**

### **`Dockerfile`**

Builds Streamlit app image:

* Installs dependencies
* Runs `streamlit_app.py`

### **`requirements.txt`**

Used ONLY for:

* Streamlit Cloud deployment
* Docker builds requiring pip
* CI/CD pipelines

UV handles deps locally — this file is optional.

---

# ☸️ **`kubernetes/` — Deployment Manifests**

### **`deployment.yaml`**

K8s deployment for the app:

* Pod replica count
* Container image
* Environment variables
* Resource limits

### **`service.yaml`**

Exposes the app:

* NodePort / LoadBalancer
* Internal cluster communication

### **`monitoring.yaml`**

Grafana + Prometheus monitoring:

* CPU, memory, pods
* Alerts
* Dashboards

---

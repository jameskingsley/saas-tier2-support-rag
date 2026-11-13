import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    # LLM Provider
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # Qdrant Cloud / Local Configuration
    qdrant_path: str = os.getenv("QDRANT_PATH", "./data/qdrant_db")
    qdrant_url: str = os.getenv("QDRANT_URL", "")
    qdrant_api_key: str = os.getenv("QDRANT_API_KEY", "")
    collection_name: str = os.getenv("COLLECTION_NAME", "tier2_support_kb")

    # Local ONNX Models
    dense_model: str = os.getenv("DENSE_MODEL", "BAAI/bge-small-en-v1.5")
    sparse_model: str = os.getenv("SPARSE_MODEL", "Qdrant/bm25")
    reranker_model: str = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

    # Langfuse Observability
    langfuse_public_key: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    langfuse_secret_key: str = os.getenv("LANGFUSE_SECRET_KEY", "")
    langfuse_base_url: str = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
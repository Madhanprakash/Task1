from functools import lru_cache
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class Settings:
    app_name = "Knowledge Graph RAG API"
    api_key: str = os.getenv("ADMIN_API_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chat_model: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "document_chunks")
    cors_origins: list[str] = [item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",") if item.strip()]
    upload_dir: Path = Path(os.getenv("UPLOAD_DIR", "data/uploads"))


@lru_cache
def get_settings() -> Settings:
    return Settings()

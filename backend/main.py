import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api import routes
from backend.core.config import get_settings
from backend.services.knowledge_base import KnowledgeBaseService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    routes.service = KnowledgeBaseService()
    routes.service.initialize()
    yield
    routes.service.close()


settings = get_settings()
app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=False, allow_methods=["GET", "POST", "DELETE"], allow_headers=["Content-Type", "X-API-Key"])
app.include_router(routes.router)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}

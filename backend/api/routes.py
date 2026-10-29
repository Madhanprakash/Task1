from pathlib import Path
from fastapi import APIRouter, Depends, File, Header, HTTPException, Query, UploadFile, status
from backend.core.config import get_settings
from backend.schemas.models import ChatRequest, ChatResponse, DocumentSummary, GraphQueryRequest, GraphResponse, PageResponse, SearchRequest, Source
from backend.services.knowledge_base import KnowledgeBaseService

router = APIRouter(prefix="/api")
service: KnowledgeBaseService | None = None


def get_service() -> KnowledgeBaseService:
    if service is None:
        raise HTTPException(503, "Knowledge base is unavailable.")
    return service


def require_admin(x_api_key: str = Header(default="")) -> None:
    configured = get_settings().api_key
    if not configured or x_api_key != configured:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Valid X-API-Key required.")


@router.post("/documents/upload", dependencies=[Depends(require_admin)])
async def upload_document(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(422, "A filename is required.")
    content = await file.read()
    if not content or len(content) > 25 * 1024 * 1024:
        raise HTTPException(413, "Files must be between 1 byte and 25 MB.")
    directory = get_settings().upload_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / Path(file.filename).name
    path.write_bytes(content)
    return {"filename": path.name, "status": "uploaded"}


@router.post("/documents/ingest", dependencies=[Depends(require_admin)])
async def ingest_document(filename: str, kb: KnowledgeBaseService = Depends(get_service)) -> dict:
    path = get_settings().upload_dir / Path(filename).name
    if not path.is_file():
        raise HTTPException(404, "Uploaded file not found.")
    try:
        return kb.ingest(path.name, path.read_bytes())
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/documents", response_model=PageResponse)
def list_documents(offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100), kb: KnowledgeBaseService = Depends(get_service)) -> PageResponse:
    items, total = kb.graph.list_documents(offset, limit)
    return PageResponse(items=[DocumentSummary(**item) for item in items], total=total, offset=offset, limit=limit)


@router.get("/documents/{document_id}", response_model=DocumentSummary)
def get_document(document_id: str, kb: KnowledgeBaseService = Depends(get_service)) -> DocumentSummary:
    items, _ = kb.graph.list_documents(0, 1000)
    for item in items:
        if item["id"] == document_id:
            return DocumentSummary(**item)
    raise HTTPException(404, "Document not found.")


@router.delete("/documents/{document_id}", dependencies=[Depends(require_admin)], status_code=204)
def delete_document(document_id: str, kb: KnowledgeBaseService = Depends(get_service)) -> None:
    kb.vectors.delete_document(document_id)
    if not kb.graph.delete_document(document_id):
        raise HTTPException(404, "Document not found.")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, kb: KnowledgeBaseService = Depends(get_service)) -> ChatResponse:
    answer, sources = kb.answer(request.question, request.top_k, request.history)
    return ChatResponse(answer=answer, sources=[Source(**item) for item in sources])


@router.post("/search", response_model=list[Source])
def search(request: SearchRequest, kb: KnowledgeBaseService = Depends(get_service)) -> list[Source]:
    return [Source(**item) for item in kb.search(request.query, request.top_k)]


@router.post("/graph/query", response_model=GraphResponse)
def graph_query(request: GraphQueryRequest, kb: KnowledgeBaseService = Depends(get_service)) -> GraphResponse:
    return GraphResponse(**kb.graph.graph(request.limit))


@router.get("/graph", response_model=GraphResponse)
def graph(limit: int = Query(50, ge=1, le=200), kb: KnowledgeBaseService = Depends(get_service)) -> GraphResponse:
    return GraphResponse(**kb.graph.graph(limit))

from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from backend.ingestion.chunker import chunk_pages
from backend.ingestion.parsers import parse_document
from backend.services.llm import LLMService
from backend.vector_store.qdrant_store import QdrantStore
from backend.graph_store.neo4j_store import Neo4jStore


class KnowledgeBaseService:
    def __init__(self) -> None:
        self.llm = LLMService()
        self.vectors = QdrantStore()
        self.graph = Neo4jStore()

    def initialize(self) -> None:
        self.vectors.initialize()
        self.graph.initialize()

    def ingest(self, filename: str, content: bytes) -> dict:
        checksum = sha256(content).hexdigest()
        document_id = checksum
        pages = parse_document(filename, content)
        chunks = chunk_pages(pages)
        embeddings = self.llm.embed([chunk["text"] for chunk in chunks])
        now = datetime.now(timezone.utc).isoformat()
        document = {"id": document_id, "filename": filename, "checksum": checksum, "status": "indexed", "created_at": now, "updated_at": now}
        graph_chunks, points, facts = [], [], []
        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{document_id}:{index}"
            record = {"id": chunk_id, **chunk, "index": index}
            graph_chunks.append(record)
            points.append({"id": str(uuid4()), "embedding": embedding, "payload": {"document_id": document_id, "chunk_id": chunk_id, "filename": filename, **chunk}})
            extracted = self.llm.extract_facts(chunk["text"])
            facts.append({"chunk_id": chunk_id, **extracted})
        self.vectors.upsert(points)
        self.graph.save_document(document, graph_chunks, facts)
        return {"document_id": document_id, "filename": filename, "status": "indexed", "chunk_count": len(chunks)}

    def search(self, query: str, top_k: int) -> list[dict]:
        return self.vectors.search(self.llm.embed([query])[0], top_k)

    def answer(self, question: str, top_k: int, history: list[dict[str, str]]) -> tuple[str, list[dict]]:
        sources = self.search(question, top_k)
        return self.llm.answer(question, sources, history), sources

    def close(self) -> None:
        self.graph.driver.close()

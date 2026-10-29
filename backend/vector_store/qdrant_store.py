from qdrant_client import QdrantClient, models
from backend.core.config import get_settings


class QdrantStore:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.client = QdrantClient(url=self.settings.qdrant_url)

    def initialize(self) -> None:
        if not self.client.collection_exists(self.settings.qdrant_collection):
            self.client.create_collection(self.settings.qdrant_collection, vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE))

    def upsert(self, points: list[dict]) -> None:
        self.client.upsert(self.settings.qdrant_collection, points=[models.PointStruct(id=x["id"], vector=x["embedding"], payload=x["payload"]) for x in points], wait=True)

    def search(self, embedding: list[float], limit: int) -> list[dict]:
        hits = self.client.query_points(self.settings.qdrant_collection, query=embedding, limit=limit).points
        return [{**hit.payload, "score": hit.score} for hit in hits]

    def delete_document(self, document_id: str) -> None:
        self.client.delete(self.settings.qdrant_collection, models.FilterSelector(filter=models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])), wait=True)

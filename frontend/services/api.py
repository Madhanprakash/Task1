import os
import httpx


class APIClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("BACKEND_URL", "http://localhost:8000/api").rstrip("/")
        self.headers = {"X-API-Key": os.getenv("ADMIN_API_KEY", "")}

    def _request(self, method: str, path: str, **kwargs):
        response = httpx.request(method, f"{self.base_url}{path}", headers=self.headers, timeout=120, **kwargs)
        response.raise_for_status()
        return response.json() if response.content else None

    def documents(self): return self._request("GET", "/documents")
    def chat(self, question, history): return self._request("POST", "/chat", json={"question": question, "history": history})
    def upload(self, file): return self._request("POST", "/documents/upload", files={"file": (file.name, file.getvalue(), file.type or "application/octet-stream")})
    def ingest(self, filename): return self._request("POST", "/documents/ingest", params={"filename": filename})
    def search(self, query): return self._request("POST", "/search", json={"query": query})

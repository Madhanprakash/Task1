# Knowledge Graph RAG

A document knowledge base with a FastAPI backend, Qdrant vector search, Neo4j knowledge graph, and Streamlit frontend. It accepts PDF, DOCX, TXT, Markdown, and CSV uploads, preserving chunk/page provenance so answers can show evidence.

## Requirements

- Python 3.10 or newer
- Neo4j 5.26+ and Qdrant (Docker Compose provides both)
- An OpenAI API key

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`. `NEO4J_DATABASE` defaults to `neo4j`; `OPENAI_CHAT_MODEL` defaults to `gpt-4o-mini`.
4. Start the backend:

   ```powershell
   uvicorn backend.main:app --reload
   ```

5. In another terminal, start the frontend:

   ```powershell
   streamlit run frontend/app.py
   ```

6. Open `http://localhost:8501`; API documentation is at `http://localhost:8000/docs`.

The app also reads credentials from `.streamlit/secrets.toml` when the corresponding environment variable is not set. Do not commit local credentials. Each upload is limited to 10 MB. Scanned PDFs need OCR before upload. Duplicate extracted text is skipped.

## Run with Docker

1. Create a `.env` file in the project directory:

   ```dotenv
   OPENAI_API_KEY=your-openai-api-key
   NEO4J_PASSWORD=choose-a-password-at-least-8-characters-long
   ADMIN_API_KEY=generate-a-long-random-value
   ```

   `NEO4J_DATABASE` and `OPENAI_CHAT_MODEL` are optional; they default to `neo4j` and `gpt-4o-mini`.
2. Start the app and Neo4j database:

   ```powershell
   docker compose up --build
   ```

3. Open <http://localhost:8501>. The API is available internally to the frontend and on <http://localhost:8000/docs> for development. Neo4j and Qdrant ports are deliberately not published.

Use `docker compose down` to stop the services. Neo4j data is retained in the `neo4j_data` volume; use `docker compose down -v` only when you want to delete that stored data.

The `.dockerignore` file excludes local credentials and development files from the image.

## Architecture and data model

- FastAPI is the only component that connects to data stores and OpenAI.
- Qdrant stores embeddings and chunk metadata for semantic retrieval.
- Neo4j stores `Document`, `Chunk`, and normalized `Entity` nodes plus `CONTAINS`, `MENTIONS`, and provenance-aware `RELATED_TO` relationships.
- Streamlit calls FastAPI only; administrative upload/index/delete requests require `X-API-Key`.

The vector index uses OpenAI's `text-embedding-3-small` model (1536 dimensions). Keep this model unchanged unless you also update the Qdrant collection dimensions.

## Verification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
docker compose up --build
```

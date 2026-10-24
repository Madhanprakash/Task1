# Document knowledge base

A Streamlit document-chat app that extracts text from PDF, DOCX, TXT, and Markdown files, then stores the extracted text, vector embeddings, and knowledge graph in Neo4j. Questions use vector search and graph relationships to retrieve relevant passages; answers cite their source documents.

## Requirements

- Python 3.10 or newer
- A Neo4j 5.18+ database with vector-index support
- An OpenAI API key

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`, `NEO4J_URI`, `NEO4J_USERNAME`, and `NEO4J_PASSWORD`. `NEO4J_DATABASE` defaults to `neo4j`; `OPENAI_CHAT_MODEL` defaults to `gpt-4o-mini`.
4. Start the app:

   ```powershell
   streamlit run streamlit_app.py
   ```

5. Upload documents in the sidebar, select **Index documents**, then ask questions in the chat.

The app also reads credentials from `.streamlit/secrets.toml` when the corresponding environment variable is not set. Do not commit local credentials. Each upload is limited to 10 MB. Scanned PDFs need OCR before upload. Duplicate extracted text is skipped.

## Run with Docker

1. Create a `.env` file in the project directory:

   ```dotenv
   OPENAI_API_KEY=your-openai-api-key
   NEO4J_PASSWORD=choose-a-password-at-least-8-characters-long
   ```

   `NEO4J_DATABASE` and `OPENAI_CHAT_MODEL` are optional; they default to `neo4j` and `gpt-4o-mini`.
2. Start the app and Neo4j database:

   ```powershell
   docker compose up --build
   ```

3. Open <http://localhost:8501>. Neo4j Browser is available at <http://localhost:7474>.

Use `docker compose down` to stop the services. Neo4j data is retained in the `neo4j_data` volume; use `docker compose down -v` only when you want to delete that stored data.

The `.dockerignore` file excludes local credentials and development files from the image.

## Neo4j data model

- `Document` nodes store the extracted document text and filename.
- `Chunk` nodes store passage text and OpenAI embeddings, with `PART_OF` links to their document.
- `Entity` nodes and `RELATED_TO` relationships store extracted knowledge-graph facts; `MENTIONS` links entities to source passages.
- A Neo4j vector index retrieves passages, and linked graph facts are included in the answer context.

The vector index uses OpenAI's `text-embedding-3-small` model (1536 dimensions). Keep this model unchanged unless you also update the vector-index dimensions.

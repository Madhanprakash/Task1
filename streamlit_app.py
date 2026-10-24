import os
from pathlib import Path
from zipfile import BadZipFile

import streamlit as st
from docx.opc.exceptions import PackageNotFoundError
from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError
from openai import OpenAI, OpenAIError
from pypdf.errors import PdfReadError

from rag import KnowledgeBase, embed_texts, ingest_document, stream_answer

load_dotenv()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value.strip()
    if Path(".streamlit/secrets.toml").is_file():
        secret_value = st.secrets.get(name, default)
        return str(secret_value).strip()
    return default


@st.cache_resource
def get_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


@st.cache_resource
def get_knowledge_base(
    uri: str, username: str, password: str, database: str
) -> KnowledgeBase:
    driver = GraphDatabase.driver(uri, auth=(username, password))
    driver.verify_connectivity()
    knowledge_base = KnowledgeBase(driver, database)
    knowledge_base.initialize()
    return knowledge_base


st.set_page_config(page_title="Document knowledge base", page_icon=":material/menu_book:")
st.title("Document knowledge base")
st.caption("Ask questions grounded in your documents, with vector and knowledge-graph retrieval.")

api_key = get_setting("OPENAI_API_KEY")
neo4j_uri = get_setting("NEO4J_URI")
neo4j_username = get_setting("NEO4J_USERNAME")
neo4j_password = get_setting("NEO4J_PASSWORD")
neo4j_database = get_setting("NEO4J_DATABASE", "neo4j")
chat_model = get_setting("OPENAI_CHAT_MODEL", "gpt-4o-mini")
required_settings = {
    "OPENAI_API_KEY": api_key,
    "NEO4J_URI": neo4j_uri,
    "NEO4J_USERNAME": neo4j_username,
    "NEO4J_PASSWORD": neo4j_password,
}
missing_settings = [name for name, value in required_settings.items() if not value]
ready = not missing_settings

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Knowledge base")
    st.caption("Stored in Neo4j")
    uploads = st.file_uploader(
        "Add documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
        help="Upload PDF, DOCX, TXT, or Markdown files. Each file is limited to 10 MB.",
    )

    if missing_settings:
        st.warning(
            "Configure these settings to connect the app: "
            + ", ".join(f"`{name}`" for name in missing_settings),
            title="Connection settings required",
        )
    else:
        st.success("OpenAI and Neo4j settings are configured.")

    if st.button(
        "Index documents",
        icon=":material/upload_file:",
        type="primary",
        width="stretch",
        disabled=not ready or not uploads,
    ):
        try:
            client = get_openai_client(api_key)
            knowledge_base = get_knowledge_base(
                neo4j_uri, neo4j_username, neo4j_password, neo4j_database
            )
            indexed_count = 0
            skipped_count = 0
            chunk_count = 0
            with st.status("Reading and indexing documents", expanded=True) as status:
                for uploaded_file in uploads:
                    if uploaded_file.size > MAX_UPLOAD_BYTES:
                        st.error(f"{uploaded_file.name} exceeds the 10 MB upload limit.")
                        continue
                    try:
                        added, chunks = ingest_document(
                            client,
                            knowledge_base,
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            chat_model,
                        )
                    except (
                        BadZipFile,
                        Neo4jError,
                        OpenAIError,
                        OSError,
                        PackageNotFoundError,
                        PdfReadError,
                        RuntimeError,
                        ValueError,
                    ) as error:
                        st.error(f"Could not index {uploaded_file.name}: {error}")
                        continue
                    if added:
                        indexed_count += 1
                        chunk_count += chunks
                        st.write(f"Indexed {uploaded_file.name} ({chunks} passages).")
                    else:
                        skipped_count += 1
                        st.write(f"Skipped {uploaded_file.name}: its text is already stored.")
                status.update(label="Document indexing finished", state="complete")
            st.success(
                f"Indexed {indexed_count} document(s) and {chunk_count} passages; "
                f"skipped {skipped_count} duplicate(s)."
            )
        except (Neo4jError, OpenAIError, OSError, RuntimeError, ValueError) as error:
            st.error(f"Could not connect to the knowledge base: {error}")

    if ready:
        try:
            knowledge_base = get_knowledge_base(
                neo4j_uri, neo4j_username, neo4j_password, neo4j_database
            )
            st.metric("Documents stored", knowledge_base.document_count())
        except (Neo4jError, OSError, RuntimeError, ValueError) as error:
            st.error(f"Neo4j is not ready: {error}")

st.subheader("Chat")
if not ready:
    st.info("Add the required API and Neo4j settings to enable document indexing and chat.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources", [])
        if sources:
            with st.expander(f"Sources ({len(sources)})"):
                for index, source in enumerate(sources, start=1):
                    st.markdown(f"**[{index}] {source['source']}**")
                    st.write(source["text"])
                    if source["graph_facts"]:
                        st.caption("Knowledge graph: " + "; ".join(source["graph_facts"]))

prompt = st.chat_input(
    "Ask a question about your documents",
    disabled=not ready,
    submit_mode="disable",
)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            client = get_openai_client(api_key)
            knowledge_base = get_knowledge_base(
                neo4j_uri, neo4j_username, neo4j_password, neo4j_database
            )
            embedding = embed_texts(client, [prompt])[0]
            sources = knowledge_base.search(embedding)
            response = st.write_stream(
                stream_answer(
                    client,
                    chat_model,
                    prompt,
                    sources,
                    st.session_state.messages[:-1],
                )
            )
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "sources": sources}
            )
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for index, source in enumerate(sources, start=1):
                        st.markdown(f"**[{index}] {source['source']}**")
                        st.write(source["text"])
                        if source["graph_facts"]:
                            st.caption("Knowledge graph: " + "; ".join(source["graph_facts"]))
            else:
                st.caption("No matching document passages were found.")
        except (Neo4jError, OpenAIError, OSError, RuntimeError, ValueError) as error:
            st.error(f"Could not answer this question: {error}")

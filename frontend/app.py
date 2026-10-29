import streamlit as st
from frontend.services.api import APIClient

st.set_page_config(page_title="Knowledge Graph RAG", page_icon=":material/account_tree:", layout="wide")
if "messages" not in st.session_state: st.session_state.messages = []
client = APIClient()

st.title("Knowledge Graph RAG")
st.caption("Ask grounded questions across your documents. Answers include source evidence.")
with st.sidebar:
    st.header("Knowledge base")
    uploads = st.file_uploader("Add documents", type=["pdf", "docx", "txt", "md", "csv"], accept_multiple_files=True)
    if st.button("Upload and index", type="primary", width="stretch", disabled=not uploads):
        try:
            with st.status("Indexing documents…", expanded=True) as status:
                for file in uploads:
                    client.upload(file)
                    client.ingest(file.name)
                    st.write(f"Indexed {file.name}")
                status.update(label="Indexing complete", state="complete")
        except Exception as exc: st.error(f"Indexing failed: {exc}")
    if st.button("New chat", width="stretch"): st.session_state.messages = []; st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"): st.markdown(question)
    with st.chat_message("assistant"):
        try:
            result = client.chat(question, st.session_state.messages[:-1])
            st.markdown(result["answer"])
            with st.expander("Sources"):
                for source in result["sources"]:
                    st.caption(f"{source['filename']} · page {source.get('page') or 'n/a'} · relevance {source.get('score', 0):.2f}")
                    st.write(source["text"])
            st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
        except Exception as exc: st.error(f"Unable to answer: {exc}")

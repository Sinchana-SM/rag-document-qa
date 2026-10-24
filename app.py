"""Streamlit UI: upload docs -> build index -> chat with sources."""
import streamlit as st

from src.config import SUPPORTED_EXTENSIONS, UPLOAD_DIR
from src.loader import load_file
from src.qa import answer_stream
from src.vectorstore import add_to_index, index_exists, load_index, reset_index

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄", layout="wide")
st.title("📄 Intelligent Document Q&A (Local AI)")
st.caption("Runs with local Ollama + local Hugging Face embeddings. No OpenAI API key required.")


@st.cache_resource(show_spinner=False)
def get_store_cached(version: int):
    """Cached FAISS handle; bump `version` to force reload after indexing."""
    return load_index()


# ---------- session state ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("index_version", 0)
st.session_state.setdefault("indexed_files", set())

# ---------- sidebar: ingestion ----------
with st.sidebar:
    st.header("1. Upload documents")
    files = st.file_uploader(
        "PDF, DOCX, TXT or MD",
        type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
    )

    if st.button("Build / update index", type="primary", disabled=not files):
        new_files = [f for f in files if f.name not in st.session_state.indexed_files]
        if not new_files:
            st.info("These files are already indexed.")
        else:
            with st.spinner("Reading, chunking and embedding..."):
                all_docs = []
                for f in new_files:
                    path = UPLOAD_DIR / f.name
                    path.write_bytes(f.getbuffer())
                    try:
                        all_docs.extend(load_file(path))
                        st.session_state.indexed_files.add(f.name)
                    except Exception as e:
                        st.warning(f"Skipped {f.name}: {e}")
                if all_docs:
                    try:
                        add_to_index(all_docs)
                        st.session_state.index_version += 1
                        st.success(f"Indexed {len(new_files)} file(s).")
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")

    st.divider()
    st.caption(f"Index status: {'✅ ready' if index_exists() else '⚠️ empty'}")

    c1, c2 = st.columns(2)
    if c1.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if c2.button("Reset index"):
        reset_index()
        st.session_state.indexed_files = set()
        st.session_state.index_version += 1
        st.session_state.messages = []
        st.rerun()

# ---------- chat ----------
store = get_store_cached(st.session_state.index_version)

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            with st.expander("Sources"):
                for s in m["sources"]:
                    st.markdown(f"**{s['label']}**")
                    st.caption(s["text"])

if question := st.chat_input("Ask a question about your documents..."):
    if store is None:
        st.warning("Upload documents and click 'Build / update index' first.")
        st.stop()

    history = list(st.session_state.messages)  # history BEFORE this question
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            stream, docs = answer_stream(store, question, history)
            answer = st.write_stream(stream)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

        sources = []
        for d in docs:
            page = d.metadata.get("page")
            label = d.metadata.get("source", "unknown") + (
                f" - page {page + 1}" if page is not None else ""
            )
            sources.append({"label": label, "text": d.page_content[:500] + "..."})
        with st.expander("Sources"):
            for s in sources:
                st.markdown(f"**{s['label']}**")
                st.caption(s["text"])

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

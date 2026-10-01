"""Web UI:  streamlit run app.py

Sidebar: upload files, optional git repo, rebuild index, manage files.
Main:    chat with history, confidence badge, expandable cited sources.
"""
import gc
from pathlib import Path

import streamlit as st

from rag import config
from rag.generation import generate_answer

st.set_page_config(page_title="Tech Q&A Bot", page_icon="🔎", layout="wide")

DOCS_DIR = Path(config.DEFAULT_DOCS_DIR)
PDF_DIR = Path(config.DEFAULT_PDF_DIR)
CODE_DIR = Path(config.DEFAULT_CODE_DIR)

DOC_EXT = {".md", ".txt", ".html", ".htm"}
PDF_EXT = {".pdf"}
CODE_EXT = {".py", ".js", ".ts", ".java", ".go", ".rs", ".c", ".cpp", ".h", ".cs", ".rb", ".php"}
TARGETS = [(DOC_EXT, DOCS_DIR), (PDF_EXT, PDF_DIR), (CODE_EXT, CODE_DIR)]
UPLOAD_TYPES = sorted(e.lstrip(".") for e in DOC_EXT | PDF_EXT | CODE_EXT)
CONF_COLOR = {"high": "green", "medium": "orange", "low": "red"}

# ---------- session state ----------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("dirty", False)
st.session_state.setdefault("flash", None)


# ---------- helpers ----------
@st.cache_resource(show_spinner="Loading indices and models...")
def get_retriever():
    from rag.retrieval import HybridRetriever
    return HybridRetriever()


def target_dir(filename: str):
    ext = Path(filename).suffix.lower()
    for exts, folder in TARGETS:
        if ext in exts:
            return folder
    return None


def save_uploads(files) -> int:
    saved = 0
    for f in files:
        folder = target_dir(f.name)
        if folder is None:
            continue
        folder.mkdir(parents=True, exist_ok=True)
        (folder / Path(f.name).name).write_bytes(f.getbuffer())
        saved += 1
    return saved


def list_files():
    found = []
    for folder in (DOCS_DIR, PDF_DIR, CODE_DIR):
        if folder.exists():
            found += [p for p in sorted(folder.rglob("*")) if p.is_file()]
    return found


def rebuild_index(repo_url: str) -> int:
    from rag.chunking import chunk_all
    from rag.indexing import build_all_indices
    from rag.ingest import ingest_all

    get_retriever.clear()  # release index handles before rebuild (Windows file locks)
    gc.collect()
    with st.status("Building index...", expanded=True) as status:
        st.write("Reading files")
        raw = ingest_all(
            repo_url=repo_url or None,
            docs_folder=str(DOCS_DIR),
            pdf_folder=str(PDF_DIR),
            code_folder=str(CODE_DIR),
        )
        st.write("Chunking")
        chunks = chunk_all(raw)
        st.write(f"Indexing {len(chunks)} chunks")
        build_all_indices(chunks)
        status.update(label=f"Index ready: {len(chunks)} chunks", state="complete", expanded=False)
    st.session_state.dirty = False
    return len(chunks)


def render_answer(msg: dict):
    st.markdown(msg["answer"])
    conf = msg["confidence"]
    st.markdown(f"Confidence: :{CONF_COLOR.get(conf, 'gray')}[**{conf}**]")
    if msg["sources"]:
        st.markdown("**Sources**")
        for s in msg["sources"]:
            with st.expander(f"[{s['id']}] {Path(s['source']).name}"):
                st.code(s["text"], language="markdown")
                st.caption(f"{s['source']}  ·  rerank score {s['score']:.3f}")


# ---------- sidebar ----------
with st.sidebar:
    st.header("📁 Knowledge base")

    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = None

    uploads = st.file_uploader(
        "Upload files",
        type=UPLOAD_TYPES,
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        help="Docs (.md .txt .html), PDFs (text-based) and source code.",
    )
    repo_url = st.text_input("Git repo URL (optional)", placeholder="https://github.com/org/repo")

    if st.session_state.dirty:
        st.warning("Files changed. Rebuild the index to apply.")

    if st.button("Build index", type="primary", use_container_width=True):
        try:
            n_saved = save_uploads(uploads or [])
            n_chunks = rebuild_index(repo_url.strip())
            st.session_state.flash = f"Saved {n_saved} file(s). Indexed {n_chunks} chunks."
            st.session_state.uploader_key += 1
            st.rerun()
        except Exception as e:
            st.error(f"Build failed: {e}")

    files = list_files()
    with st.expander(f"Files in knowledge base ({len(files)})"):
        if files:
            doomed = st.multiselect("Select files to delete", files, format_func=lambda p: p.name)
            if st.button("Delete selected", disabled=not doomed):
                for p in doomed:
                    p.unlink(missing_ok=True)
                st.session_state.dirty = True
                st.rerun()
        else:
            st.caption("No files yet.")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- main ----------
st.title("🔎 Multi-Source Technical Q&A Bot")
st.caption("Answers grounded in your docs, PDFs and code, with cited sources.")

try:
    retriever = get_retriever()
except RuntimeError as e:
    st.info("No index yet. Upload files in the sidebar, then click **Build index**.")
    st.caption(str(e))
    st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["content"])
        else:
            render_answer(m)

if question := st.chat_input("Ask about your docs, code or PDFs"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                chunks = retriever.retrieve(question, verbose=False)
                result = generate_answer(question, chunks)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        by_id = {c["chunk_id"]: c for c in chunks}
        sources = []
        for cit in result.citations:
            chunk = by_id.get(cit.chunk_id)
            sources.append({
                "id": cit.chunk_id,
                "source": cit.source,
                "text": chunk["text"] if chunk else "",
                "score": chunk.get("rerank_score", 0) if chunk else 0,
            })
        msg = {
            "role": "assistant",
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": sources,
        }
        render_answer(msg)
    st.session_state.messages.append(msg)
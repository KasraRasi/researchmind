"""
ResearchMind — Phase 4: Streamlit Chat UI
==========================================

WHAT THIS FILE DOES:
  Builds a chat interface on top of the graph we built in Phases 1-3.
  Users can:
    - Upload PDF or text documents
    - Ask questions in a chat interface
    - See answers with cited sources
    - Know whether the answer came from documents or web search

HOW TO RUN:
  streamlit run src/app.py

CORE CONCEPT — Why Streamlit?
  Streamlit turns a Python script into a web app with almost no extra
  code. Every time the user interacts (types a message, uploads a file),
  the script reruns from top to bottom. st.session_state persists data
  between reruns — like a global variable that survives page refreshes.
"""

import os
import shutil
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Project paths
DOCS_DIR = Path(__file__).parent.parent / "data" / "sample_docs"
FAISS_DIR = Path(__file__).parent.parent / "data" / "faiss_db"

# ---------------------------------------------------------------------------
# PAGE CONFIG — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------
# NOTE: st.session_state persists values between reruns.
# Without it, every message would reset the chat history.

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = FAISS_DIR.exists()

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def load_graph():
    """Import and build the graph — done here so imports stay clean."""
    from graph import build_graph
    return build_graph()


def run_ingestion_pipeline(docs_dir: Path):
    """Run the ingestion pipeline and return success bool."""
    from ingest import load_documents, split_documents, build_vector_store
    docs = load_documents(docs_dir)
    if not docs:
        return False
    chunks = split_documents(docs)
    store  = build_vector_store(chunks)
    return store is not None


def format_sources(sources: list, source_type: str) -> str:
    """Format sources list into clean readable text."""
    if not sources:
        return ""

    if source_type == "web":
        # Filter out generic "web" placeholders, keep real URLs
        real_urls = [s for s in sources if s.startswith("http")]
        if real_urls:
            return "\n".join(f"- {url}" for url in real_urls)
        return "- Web search results"
    else:
        unique = list(set(sources))
        return "\n".join(f"- {s}" for s in unique)


# ---------------------------------------------------------------------------
# SIDEBAR — document upload + ingestion
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔍 ResearchMind")
    st.caption("Multi-agent RAG powered by LangGraph")

    st.divider()

    st.subheader("Documents")

    uploaded_files = st.file_uploader(
        "Upload PDFs or text files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Index documents", type="primary", use_container_width=True):
            # Save uploaded files to docs directory
            DOCS_DIR.mkdir(parents=True, exist_ok=True)

            with st.spinner("Saving files..."):
                for file in uploaded_files:
                    dest = DOCS_DIR / file.name
                    with open(dest, "wb") as f:
                        f.write(file.getbuffer())

            # Run ingestion
            with st.spinner("Building vector store... (this may take a moment)"):
                success = run_ingestion_pipeline(DOCS_DIR)

            if success:
                st.session_state.vector_store_ready = True
                st.success(f"✅ Indexed {len(uploaded_files)} file(s)")
            else:
                st.error("Failed to index documents. Check your API key.")

    st.divider()

    # Status indicator
    if st.session_state.vector_store_ready:
        st.success("Vector store ready")
    else:
        st.warning("No documents indexed yet")

    # Show indexed files
    if DOCS_DIR.exists():
        files = list(DOCS_DIR.glob("**/*.txt")) + \
                list(DOCS_DIR.glob("**/*.pdf")) + \
                list(DOCS_DIR.glob("**/*.md"))
        if files:
            st.caption("Indexed files:")
            for f in files:
                st.caption(f"  • {f.name}")

    st.divider()

    # Clear chat button
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("Built with LangGraph + RAG")
    st.caption("Phase 4 — Portfolio project")


# ---------------------------------------------------------------------------
# MAIN CHAT UI
# ---------------------------------------------------------------------------

st.title("ResearchMind")
st.caption("Ask questions about your documents. Falls back to web search when needed.")

# Show existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Show sources if present
        if message.get("sources") and message.get("source_type"):
            source_text = format_sources(message["sources"], message["source_type"])
            if source_text:
                badge = "📄 Documents" if message["source_type"] == "documents" else "🌐 Web search"
                with st.expander(f"Sources — {badge}"):
                    st.markdown(source_text)

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):

    # Check API key
    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY not set. Add it to your .env file.")
        st.stop()

    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Run the graph
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                graph = load_graph()
                result = graph.invoke({
                    "question":       prompt,
                    "chunks":         [],
                    "grade":          "",
                    "search_results": [],
                    "source_type":    "documents",
                    "answer":         "",
                    "sources":        [],
                })

                answer      = result.get("answer", "I couldn't generate an answer.")
                sources     = result.get("sources", [])
                source_type = result.get("source_type", "documents")

                # Clean up the answer — remove the "search results do not contain"
                # disclaimer if the LLM still answered correctly
                if "search results do not contain" in answer.lower() and len(answer) > 80:
                    # LLM answered anyway after the disclaimer — keep just the answer
                    parts = answer.split("However,")
                    if len(parts) > 1:
                        answer = "However," + parts[1].strip()

                # Display answer
                st.markdown(answer)

                # Display sources
                source_text = format_sources(sources, source_type)
                if source_text:
                    badge = "📄 Documents" if source_type == "documents" else "🌐 Web search"
                    with st.expander(f"Sources — {badge}"):
                        st.markdown(source_text)

                # Save to history
                st.session_state.messages.append({
                    "role":        "assistant",
                    "content":     answer,
                    "sources":     sources,
                    "source_type": source_type,
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role":    "assistant",
                    "content": error_msg,
                })

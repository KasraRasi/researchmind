"""
ResearchMind — Production Upgrade: App v2
==========================================

CHANGES FROM v1:
  1. Graph is cached with @st.cache_resource — compiled once per session,
     not rebuilt on every message. Eliminates the biggest performance bottleneck.

  2. reset_retriever() is called after ingestion — forces the singleton
     retriever to reload from the updated FAISS index when new docs are added.

  3. State key updated — added rewritten_query to match GraphState v2.
"""

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DOCS_DIR  = Path(__file__).parent.parent / "data" / "sample_docs"
FAISS_DIR = Path(__file__).parent.parent / "data" / "faiss_db"

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="ResearchMind",
    page_icon="🔍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store_ready" not in st.session_state:
    st.session_state.vector_store_ready = FAISS_DIR.exists()

if "keys_set" not in st.session_state:
    st.session_state.keys_set = bool(os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# CACHED GRAPH
# Design decision: @st.cache_resource caches the compiled graph across all
# reruns. Streamlit reruns the script on every interaction — without caching,
# build_graph() would recompile the LangGraph on every message sent.
# ---------------------------------------------------------------------------

@st.cache_resource
def load_graph():
    from graph import build_graph
    return build_graph()


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def set_api_keys(openai_key: str, tavily_key: str):
    os.environ["OPENAI_API_KEY"] = openai_key
    if tavily_key:
        os.environ["TAVILY_API_KEY"] = tavily_key
    st.session_state.keys_set = True


def run_ingestion_pipeline(docs_dir: Path) -> bool:
    from ingest import load_documents, split_documents, build_vector_store
    docs = load_documents(docs_dir)
    if not docs:
        return False
    chunks = split_documents(docs)
    store  = build_vector_store(chunks)
    if store:
        # Reset the singleton retriever so it reloads the updated FAISS index
        from graph import reset_retriever
        reset_retriever()
    return store is not None


def format_sources(sources: list, source_type: str) -> str:
    if not sources:
        return ""
    if source_type == "web":
        real_urls = [s for s in sources if s.startswith("http")]
        return "\n".join(f"- {url}" for url in real_urls) if real_urls else "- Web search results"
    return "\n".join(f"- {s}" for s in list(set(sources)))


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔍 ResearchMind")
    st.caption("Multi-agent RAG powered by LangGraph")
    st.divider()

    st.subheader("🔑 API Keys")

    if not st.session_state.keys_set:
        st.caption("Keys stay in your browser session only — never stored.")

        openai_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...",
                                   help="Get yours at platform.openai.com")
        tavily_key = st.text_input("Tavily API Key", type="password", placeholder="tvly-...",
                                   help="Free key at tavily.com — needed for web search fallback")

        if st.button("Save keys", type="primary", use_container_width=True):
            if openai_key:
                set_api_keys(openai_key, tavily_key)
                if not tavily_key:
                    st.warning("Tavily key missing — web search fallback disabled")
                else:
                    st.success("✅ Keys saved for this session")
                st.rerun()
            else:
                st.error("OpenAI key is required")

        st.divider()
        st.caption("💡 Get a free OpenAI key at [platform.openai.com](https://platform.openai.com)")
        st.caption("💡 Get a free Tavily key at [tavily.com](https://tavily.com)")

    else:
        st.success("✅ API keys active")
        if st.button("Change keys", use_container_width=True):
            st.session_state.keys_set = False
            os.environ.pop("OPENAI_API_KEY", None)
            os.environ.pop("TAVILY_API_KEY", None)
            st.cache_resource.clear()
            st.rerun()

    st.divider()

    if st.session_state.keys_set:
        st.subheader("📄 Documents")

        uploaded_files = st.file_uploader(
            "Upload PDFs or text files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
        )

        if uploaded_files:
            if st.button("Index documents", type="primary", use_container_width=True):
                DOCS_DIR.mkdir(parents=True, exist_ok=True)
                with st.spinner("Saving files..."):
                    for file in uploaded_files:
                        with open(DOCS_DIR / file.name, "wb") as f:
                            f.write(file.getbuffer())

                with st.spinner("Building vector store..."):
                    success = run_ingestion_pipeline(DOCS_DIR)

                if success:
                    st.session_state.vector_store_ready = True
                    st.success(f"✅ Indexed {len(uploaded_files)} file(s)")
                else:
                    st.error("Failed to index. Check your OpenAI key.")

        st.divider()

        if st.session_state.vector_store_ready:
            st.success("Vector store ready")
        else:
            st.warning("No documents indexed yet")

        if DOCS_DIR.exists():
            files = (list(DOCS_DIR.glob("**/*.txt")) +
                     list(DOCS_DIR.glob("**/*.pdf")) +
                     list(DOCS_DIR.glob("**/*.md")))
            if files:
                st.caption("Indexed files:")
                for f in files:
                    st.caption(f"  • {f.name}")

        st.divider()

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Built with LangGraph + RAG")
    st.caption("[GitHub](https://github.com/KasraRasi/researchmind)")


# ---------------------------------------------------------------------------
# MAIN CHAT UI
# ---------------------------------------------------------------------------

st.title("ResearchMind 🔍")
st.caption("Ask questions about your documents. Falls back to web search when needed.")

if not st.session_state.keys_set:
    st.info("👈 Enter your API keys in the sidebar to get started.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources") and message.get("source_type"):
            source_text = format_sources(message["sources"], message["source_type"])
            if source_text:
                badge = "📄 Documents" if message["source_type"] == "documents" else "🌐 Web search"
                with st.expander(f"Sources — {badge}"):
                    st.markdown(source_text)

if prompt := st.chat_input("Ask a question about your documents..."):

    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                graph  = load_graph()
                result = graph.invoke({
                    "question":        prompt,
                    "rewritten_query": "",
                    "chunks":          [],
                    "grade":           "",
                    "search_results":  [],
                    "source_type":     "documents",
                    "answer":          "",
                    "sources":         [],
                })

                answer      = result.get("answer", "I couldn't generate an answer.")
                sources     = result.get("sources", [])
                source_type = result.get("source_type", "documents")

                if "search results do not contain" in answer.lower() and len(answer) > 80:
                    parts = answer.split("However,")
                    if len(parts) > 1:
                        answer = "However," + parts[1].strip()

                st.markdown(answer)

                source_text = format_sources(sources, source_type)
                if source_text:
                    badge = "📄 Documents" if source_type == "documents" else "🌐 Web search"
                    with st.expander(f"Sources — {badge}"):
                        st.markdown(source_text)

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
                    "role": "assistant", "content": error_msg
                })
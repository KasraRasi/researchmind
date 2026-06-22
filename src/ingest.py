"""
ResearchMind — Phase 1: Document Ingestion Pipeline
=====================================================

WHAT THIS FILE DOES:
  1. Loads documents from a folder (PDFs, .txt, .md files)
  2. Splits them into smaller chunks (so the LLM can process them)
  3. Converts each chunk into an embedding (a list of numbers = meaning)
  4. Stores everything in a FAISS vector store (saved locally to disk)

CORE CONCEPT — Why do we chunk documents?
  LLMs have a context window limit. You can't feed a 200-page PDF into
  a single prompt. So we split the doc into small overlapping pieces,
  embed each one, and later retrieve only the most relevant pieces.

CORE CONCEPT — What is an embedding?
  A vector (list of ~1536 numbers) that captures the *meaning* of text.
  Similar meaning = similar numbers = close together in vector space.
  "dog" and "puppy" will be closer to each other than "dog" and "rocket".
  This is how semantic search works — we search by meaning, not keywords.

CORE CONCEPT — Why FAISS instead of ChromaDB?
  FAISS (Facebook AI Similarity Search) is a lightweight vector store
  that saves directly to disk as simple files. No server, no DLL issues,
  works on every OS. Perfect for local development and portfolio projects.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DOCS_DIR = Path(__file__).parent.parent / "data" / "sample_docs"
FAISS_DIR = Path(__file__).parent.parent / "data" / "faiss_db"

# chunk_size: how many characters per chunk (~250 tokens per 1000 chars)
#   Too small → chunks lose context
#   Too large → too much noise, retrieval quality drops
CHUNK_SIZE = 1000

# chunk_overlap: how many characters the next chunk repeats from the previous
#   Prevents meaning loss at chunk boundaries
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------------
# STEP 1 — Load documents
# ---------------------------------------------------------------------------

def load_documents(docs_dir: Path) -> list:
    """
    Loads all supported files from a directory.

    Each Document object has two things:
      - doc.page_content  → the raw text
      - doc.metadata      → dict with source path, page number, etc.
    """
    documents = []

    if not docs_dir.exists():
        print(f"[!] Docs folder not found: {docs_dir}")
        return documents

    loaders = [
        DirectoryLoader(str(docs_dir), glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True),
        DirectoryLoader(str(docs_dir), glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
        DirectoryLoader(str(docs_dir), glob="**/*.md",  loader_cls=TextLoader, show_progress=True),
    ]

    for loader in loaders:
        try:
            docs = loader.load()
            documents.extend(docs)
            print(f"  Loaded {len(docs)} document(s) via {loader.__class__.__name__}")
        except Exception as e:
            print(f"  Warning: {e}")

    print(f"\n✓ Total documents loaded: {len(documents)}")
    return documents


# ---------------------------------------------------------------------------
# STEP 2 — Split into chunks
# ---------------------------------------------------------------------------

def split_documents(documents: list) -> list:
    """
    Splits large documents into smaller overlapping chunks.

    RecursiveCharacterTextSplitter tries to split on paragraph breaks first
    then sentences, then words, then characters — keeping logical units together.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )

    chunks = splitter.split_documents(documents)

    print(f"✓ Split into {len(chunks)} chunks")
    print(f"  (avg chunk size: {sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)} chars)")
    return chunks


# ---------------------------------------------------------------------------
# STEP 3 — Embed and store
# ---------------------------------------------------------------------------

def build_vector_store(chunks: list) -> FAISS:
    """
    Converts chunks into embeddings and stores them with FAISS.

    FAISS.from_documents:
      1. Calls the OpenAI embeddings API on each chunk
      2. Builds an in-memory index
      3. Saves everything to disk at FAISS_DIR
    """
    print("\nBuilding embeddings — calling OpenAI API...")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
    )

    try:
        vector_store = FAISS.from_documents(chunks, embeddings)
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        vector_store.save_local(str(FAISS_DIR))
        print(f"✓ Vector store built with {len(chunks)} vectors")
        print(f"  Saved to: {FAISS_DIR}")
        return vector_store
    except Exception as e:
        print(f"ERROR during embedding: {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# STEP 4 — Load existing store
# ---------------------------------------------------------------------------

def load_vector_store() -> FAISS:
    """
    Loads an already-built FAISS store from disk.

    Embedding is expensive — only run build_vector_store() once per document
    set. After that, use this to reconnect instantly without API calls.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print(f"✓ Loaded existing vector store from {FAISS_DIR}")
    return vector_store


# ---------------------------------------------------------------------------
# STEP 5 — Test retrieval
# ---------------------------------------------------------------------------

def test_retrieval(vector_store: FAISS, query: str, k: int = 4):
    """
    Runs a similarity search and prints results.

    k=4 means return the 4 most relevant chunks.
    The retriever embeds your query and finds the nearest chunk vectors.
    Try different queries to see what comes back — bad retrieval = bad answers.
    """
    print(f"\nTest query: '{query}'")
    print("-" * 50)

    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    results = retriever.invoke(query)

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"\n[Chunk {i}] Source: {Path(source).name}")
        print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

    return results


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_ingestion():
    """Full pipeline: load → split → embed → store → test."""

    print("=" * 55)
    print("  ResearchMind — Phase 1: Document Ingestion")
    print("=" * 55)

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY not set. Check your .env file.")

    docs = load_documents(DOCS_DIR)
    if not docs:
        print(f"\n[!] No documents found in: {DOCS_DIR}")
        return None

    chunks = split_documents(docs)
    store = build_vector_store(chunks)

    if store:
        test_retrieval(store, "What is the main topic of these documents?")
        print("\n✅ Phase 1 complete! Vector store is ready.")
        print("   Next: run graph.py for Phase 2.")

    return store


if __name__ == "__main__":
    run_ingestion()

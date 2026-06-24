"""
ResearchMind — Production Upgrade: Graph v2
============================================

CHANGES FROM v1:
  1. Module-level singletons — LLM and retriever are created once at import
     time, not rebuilt on every node call. Eliminates redundant API client
     instantiation and FAISS disk reads on every message.

  2. Query rewriter node — added as the first node in the graph. Rewrites
     conversational or ambiguous user questions into clean, self-contained
     retrieval queries before hitting the vector store.

     Why this matters: users ask things like "what about the second point?"
     or "can you elaborate on that?" — these embed poorly and return garbage
     chunks. The rewriter turns them into precise retrieval queries.

  3. Tighter grader prompt — removed "be generous" instruction which caused
     the grader to almost never trigger web search fallback. Now requires
     chunks to actually contain the answer, not just be vaguely related.

GRAPH FLOW:
  rewrite → retrieve → grade → generate   → END  (documents path)
                             ↘ web_search → generate → END  (web path)
"""

from pathlib import Path
from typing import List
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_tavily import TavilySearch
from langgraph.graph import START, END, StateGraph

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

FAISS_DIR  = Path(__file__).parent.parent / "data" / "faiss_db"
RETRIEVE_K = 4

# ---------------------------------------------------------------------------
# MODULE-LEVEL SINGLETONS
# ---------------------------------------------------------------------------
# Design decision: instantiate LLM and retriever once at module load time.
# Previously get_llm() and get_retriever() were called inside every node,
# meaning each message rebuilt the OpenAI client and reloaded the FAISS
# index from disk. Moving to singletons eliminates that overhead entirely.
#
# _retriever is lazy-initialized on first use because the FAISS index
# may not exist yet when the module loads (e.g. before first ingestion).

_llm       = None
_retriever = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return _llm


def get_retriever():
    global _retriever
    if _retriever is None:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.load_local(
            str(FAISS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        _retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVE_K})
    return _retriever


def reset_retriever():
    """
    Call this after new documents are ingested to force the retriever
    to reload from the updated FAISS index.
    """
    global _retriever
    _retriever = None


# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    """
    Shared state passed between every node.

    New in v2:
      - rewritten_query: the cleaned-up version of the user's question,
        used for retrieval instead of the raw question. The raw question
        is still used for generation so the answer sounds natural.
    """
    question:        str
    rewritten_query: str            # NEW — set by rewrite node
    chunks:          List[Document]
    grade:           str
    search_results:  List[dict]
    source_type:     str
    answer:          str
    sources:         List[str]


# ---------------------------------------------------------------------------
# NODE 1 — rewrite (NEW)
# ---------------------------------------------------------------------------

def rewrite(state: GraphState) -> dict:
    """
    Rewrites the user's question into a clean retrieval query.

    Design decision: users ask conversational questions ("what about that
    second point?", "can you give more detail?", "and the limitations?").
    These embed poorly because they rely on context that isn't in the
    question itself. The rewriter makes the query self-contained.

    Examples:
      "what about the second point?"
        → "what is the second point about retrieval-augmented generation?"
      "and the limitations?"
        → "what are the limitations of large language models?"
      "What is RAG?"
        → "What is RAG?" (already clean, returned as-is)

    Reads:   state["question"]
    Updates: state["rewritten_query"]
    """
    print("\n[Node: rewrite]")
    question = state["question"]

    llm = get_llm()

    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a query rewriter for a document retrieval system. "
         "Your job is to rewrite the user's question into a clear, self-contained "
         "search query that will retrieve the most relevant document chunks.\n\n"
         "Rules:\n"
         "- Make the query specific and self-contained (no pronouns like 'it', 'that', 'this')\n"
         "- Remove conversational filler ('can you explain', 'I want to know about')\n"
         "- If the question is already a clean search query, return it as-is\n"
         "- Return ONLY the rewritten query, nothing else"),
        ("human", "Question: {question}"),
    ])

    result          = (rewrite_prompt | llm).invoke({"question": question})
    rewritten_query = result.content.strip()

    # If rewrite looks wrong (too long or empty), fall back to original
    if not rewritten_query or len(rewritten_query) > 300:
        rewritten_query = question

    print(f"  Original:  '{question}'")
    print(f"  Rewritten: '{rewritten_query}'")

    return {"rewritten_query": rewritten_query}


# ---------------------------------------------------------------------------
# NODE 2 — retrieve
# ---------------------------------------------------------------------------

def retrieve(state: GraphState) -> dict:
    """
    Searches FAISS using the rewritten query.

    Design decision: uses rewritten_query instead of question for retrieval
    to get more precise chunk matches. The raw question is preserved in
    state for the generate node which needs it to sound natural.

    Reads:   state["rewritten_query"]
    Updates: state["chunks"]
    """
    print("\n[Node: retrieve]")
    query = state.get("rewritten_query") or state["question"]
    print(f"  Searching for: '{query}'")

    retriever = get_retriever()
    chunks    = retriever.invoke(query)

    print(f"  Found {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        print(f"    Chunk {i}: {Path(source).name}")

    return {"chunks": chunks}


# ---------------------------------------------------------------------------
# NODE 3 — grade
# ---------------------------------------------------------------------------

def grade(state: GraphState) -> dict:
    """
    Uses the LLM to decide if retrieved chunks are relevant.

    Design decision: removed "be generous" from the prompt — it caused the
    grader to return "relevant" even when chunks were only loosely related,
    meaning web search fallback almost never triggered. The new prompt
    requires chunks to actually contain enough information to answer.

    Reads:   state["question"], state["chunks"]
    Updates: state["grade"]
    """
    print("\n[Node: grade]")
    question = state["question"]
    chunks   = state["chunks"]

    if not chunks:
        print("  No chunks found — marking as not_relevant")
        return {"grade": "not_relevant"}

    llm         = get_llm()
    chunks_text = "\n\n---\n\n".join(c.page_content for c in chunks)

    grader_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a relevance grader for a RAG system. "
         "Given a user question and retrieved document chunks, decide if the chunks "
         "contain sufficient information to answer the question accurately.\n\n"
         "Grade as 'relevant' ONLY if the chunks directly address the question.\n"
         "Grade as 'not_relevant' if the chunks are only loosely related, "
         "off-topic, or lack the specific details needed to answer.\n"
         "Respond with ONLY one word: 'relevant' or 'not_relevant'."),
        ("human", "Question: {question}\n\nDocument chunks:\n{chunks}"),
    ])

    result      = (grader_prompt | llm).invoke({"question": question, "chunks": chunks_text})
    grade_value = result.content.strip().lower()
    grade_value = "not_relevant" if "not_relevant" in grade_value else "relevant"

    print(f"  Grade: {grade_value}")
    return {"grade": grade_value}


# ---------------------------------------------------------------------------
# NODE 4 — web_search
# ---------------------------------------------------------------------------

def web_search(state: GraphState) -> dict:
    """
    Searches the web using Tavily. Fetch only — no generation.

    Reads:   state["question"]
    Updates: state["search_results"], state["source_type"]
    """
    print("\n[Node: web_search]")
    question = state["question"]
    print(f"  Searching web for: '{question}'")

    tavily  = TavilySearch(max_results=3)
    raw     = tavily.invoke(question)
    # New TavilySearch returns a single string, wrap it properly
    if isinstance(raw, str):
        results = [{"url": "web search", "content": raw}]
    elif isinstance(raw, list):
        results = [r if isinstance(r, dict) else {"url": "web", "content": str(r)} for r in raw]
    else:
        results = [{"url": "web", "content": str(raw)}]

    normalized = []
    for r in results:
        if isinstance(r, dict):
            normalized.append(r)
        else:
            normalized.append({"url": "web", "content": str(r)})

    print(f"  Got {len(normalized)} web results")
    for i, r in enumerate(normalized, 1):
        print(f"    Result {i}: {r.get('url', 'unknown')}")

    return {"search_results": normalized, "source_type": "web"}


# ---------------------------------------------------------------------------
# NODE 5 — generate
# ---------------------------------------------------------------------------

def generate(state: GraphState) -> dict:
    """
    Generates the final answer from whatever context is in state.

    Handles both document chunks and web results by checking source_type.
    Always uses the original question (not the rewritten query) so the
    answer sounds natural in response to what the user actually asked.

    Reads:   state["question"], state["chunks"] or state["search_results"]
    Updates: state["answer"], state["sources"]
    """
    print("\n[Node: generate]")
    question    = state["question"]
    source_type = state.get("source_type", "documents")
    llm         = get_llm()

    if source_type == "web":
        results = state["search_results"]
        context = "\n\n---\n\n".join(
            f"Source: {r.get('url', 'unknown')}\n{r.get('content', '')}"
            for r in results
        )
        sources       = [r.get("url", "unknown") for r in results]
        context_label = "web search results"
    else:
        chunks  = state["chunks"]
        context = "\n\n---\n\n".join(c.page_content for c in chunks)
        sources = list(set(
            Path(c.metadata.get("source", "unknown")).name
            for c in chunks
        ))
        context_label = "document context"

    generate_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful research assistant. Answer the question using ONLY "
         "the provided {context_label}. If it doesn't contain the answer, say so. "
         "Be concise and clear."),
        ("human", "{context_label}:\n{context}\n\nQuestion: {question}"),
    ])

    result = (generate_prompt | llm).invoke({
        "context_label": context_label,
        "context":       context,
        "question":      question,
    })

    answer = result.content.strip()
    print(f"  Answer generated ({len(answer)} chars)")
    print(f"  Sources: {sources}")

    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# CONDITIONAL EDGE
# ---------------------------------------------------------------------------

def should_search_web(state: GraphState) -> str:
    grade_value = state["grade"]
    print(f"\n[Edge: should_search_web] grade={grade_value}")

    if grade_value == "relevant":
        print("  Routing to: generate (documents path)")
        return "generate"
    else:
        print("  Routing to: web_search (fallback path)")
        return "web_search"


# ---------------------------------------------------------------------------
# BUILD THE GRAPH
# ---------------------------------------------------------------------------

def build_graph():
    """
    Graph flow:
      START → rewrite → retrieve → grade → generate   → END
                                         ↘ web_search → generate → END
    """
    graph = StateGraph(GraphState)

    graph.add_node("rewrite",    rewrite)
    graph.add_node("retrieve",   retrieve)
    graph.add_node("grade",      grade)
    graph.add_node("web_search", web_search)
    graph.add_node("generate",   generate)

    graph.add_edge(START,        "rewrite")
    graph.add_edge("rewrite",    "retrieve")
    graph.add_edge("retrieve",   "grade")

    graph.add_conditional_edges(
        "grade",
        should_search_web,
        {"generate": "generate", "web_search": "web_search"},
    )

    graph.add_edge("web_search", "generate")
    graph.add_edge("generate",   END)

    compiled = graph.compile()
    print("✓ Graph compiled successfully")
    return compiled


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_query(question: str):
    print("=" * 55)
    print("  ResearchMind v2 — Production Graph")
    print("=" * 55)
    print(f"\nQuestion: {question}\n")

    graph  = build_graph()
    result = graph.invoke({
        "question":        question,
        "rewritten_query": "",
        "chunks":          [],
        "grade":           "",
        "search_results":  [],
        "source_type":     "documents",
        "answer":          "",
        "sources":         [],
    })

    print("\n" + "=" * 55)
    print(f"ANSWER (via {result.get('source_type', 'unknown')}):")
    print(result["answer"] if result["answer"] else "[No answer generated]")
    print("\nSOURCES:")
    for s in result.get("sources", []):
        print(f"  - {s}")
    print("=" * 55)
    return result


if __name__ == "__main__":
    run_query("What is retrieval-augmented generation?")
    print("\n\n")
    run_query("What is the capital of France?")
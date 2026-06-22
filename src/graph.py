"""
ResearchMind — Phase 3: Multi-Agent Graph with Web Search Fallback
==================================================================

ARCHITECTURE:
  Every node does exactly ONE job (single responsibility principle).

  Path A — answer from documents:
    retrieve → grade → generate → END

  Path B — answer from web (fallback):
    retrieve → grade → web_search → generate → END

  The generate node handles both paths — it checks whether it has
  document chunks or web results in state and builds context accordingly.
  This makes generate reusable and each node independently testable.

NODES:
  retrieve   — searches FAISS, returns chunks
  grade      — LLM decides if chunks are relevant
  web_search — Tavily search, returns raw results (no generation)
  generate   — builds answer from whatever context is in state

EDGES:
  grade → generate   (if relevant)
  grade → web_search (if not relevant)
  web_search → generate (always)
  generate → END
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
# STATE
# ---------------------------------------------------------------------------

class GraphState(TypedDict):
    """
    Shared state passed between every node.

    source_type tells generate which context to use:
      "documents" → build context from chunks
      "web"       → build context from search_results
    """
    question:       str
    chunks:         List[Document]  # set by retrieve
    grade:          str             # set by grade
    search_results: List[dict]      # set by web_search (Tavily results)
    source_type:    str             # "documents" or "web"
    answer:         str             # set by generate
    sources:        List[str]       # set by generate


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def get_retriever():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local(
        str(FAISS_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store.as_retriever(search_kwargs={"k": RETRIEVE_K})


def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------------------------------------------------------------------------
# NODE 1 — retrieve
# ---------------------------------------------------------------------------

def retrieve(state: GraphState) -> dict:
    """
    Searches FAISS and returns the most relevant chunks.

    Single responsibility: similarity search only.
    No LLM call, no generation — just fetch chunks.

    Reads:   state["question"]
    Updates: state["chunks"]
    """
    print("\n[Node: retrieve]")
    question = state["question"]
    print(f"  Searching for: '{question}'")

    retriever = get_retriever()
    chunks    = retriever.invoke(question)

    print(f"  Found {len(chunks)} chunks")
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        print(f"    Chunk {i}: {Path(source).name}")

    return {"chunks": chunks}


# ---------------------------------------------------------------------------
# NODE 2 — grade
# ---------------------------------------------------------------------------

def grade(state: GraphState) -> dict:
    """
    Uses the LLM to decide if retrieved chunks are relevant.

    Single responsibility: relevance classification only.
    Returns "relevant" or "not_relevant" — nothing else.

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
         "You are a relevance grader. Given a question and some document chunks, "
         "decide if the chunks contain enough information to answer the question.\n"
         "Respond with ONLY one word: 'relevant' or 'not_relevant'.\n"
         "Be generous — if the chunks are even partially helpful, say 'relevant'."),
        ("human", "Question: {question}\n\nDocument chunks:\n{chunks}"),
    ])

    result      = (grader_prompt | llm).invoke({"question": question, "chunks": chunks_text})
    grade_value = result.content.strip().lower()
    grade_value = "not_relevant" if "not_relevant" in grade_value else "relevant"

    print(f"  Grade: {grade_value}")
    return {"grade": grade_value}


# ---------------------------------------------------------------------------
# NODE 3 — web_search
# ---------------------------------------------------------------------------

def web_search(state: GraphState) -> dict:
    """
    Searches the web using Tavily and stores raw results in state.

    Single responsibility: fetch web results only. No generation.
    The generate node will turn these results into an answer.

    NOTE: This is the clean separation you suggested.
    web_search just fetches data and sets source_type to "web".
    generate then handles both document and web contexts uniformly.

    Reads:   state["question"]
    Updates: state["search_results"], state["source_type"]
    """
    print("\n[Node: web_search]")
    question = state["question"]
    print(f"  Searching web for: '{question}'")

    tavily  = TavilySearch(max_results=3)
    results = tavily.invoke(question)

    normalized = []
    for r in results:
        if isinstance(r, dict):
            normalized.append(r)
        else:
            normalized.append({"url": "web", "content": str(r)})

    print(f"  Got {len(normalized)} web results")
    for i, r in enumerate(normalized, 1):
        print(f"    Result {i}: {r.get('url', 'unknown')}")

    return {
        "search_results": normalized,
        "source_type":    "web",
    }


# ---------------------------------------------------------------------------
# NODE 4 — generate
# ---------------------------------------------------------------------------

def generate(state: GraphState) -> dict:
    """
    Generates the final answer from whatever context is in state.

    Single responsibility: answer generation only.
    Handles both document chunks and web results by checking source_type.

    NOTE: This is the benefit of your refactor suggestion.
    One generate node handles both paths cleanly. It builds context
    differently depending on source_type, but the generation logic
    (prompt → LLM → answer) is shared and not duplicated.

    Reads:   state["question"], state["chunks"] or state["search_results"]
    Updates: state["answer"], state["sources"]
    """
    print("\n[Node: generate]")
    question    = state["question"]
    source_type = state.get("source_type", "documents")
    llm         = get_llm()

    # Build context and sources differently depending on where data came from
    if source_type == "web":
        results = state["search_results"]
        context = "\n\n---\n\n".join(
            f"Source: {r.get('url', 'unknown')}\n{r.get('content', '')}"
            for r in results
        )
        sources = [r.get("url", "unknown") for r in results]
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
    print(f"  Source type: {source_type}")
    print(f"  Sources: {sources}")

    return {"answer": answer, "sources": sources}


# ---------------------------------------------------------------------------
# CONDITIONAL EDGE
# ---------------------------------------------------------------------------

def should_search_web(state: GraphState) -> str:
    """
    Routes after grading:
      relevant     → generate (answer from documents)
      not_relevant → web_search (fetch web results first)

    NOTE: Notice the edge now goes to web_search, not generate,
    when not relevant. web_search then flows to generate via a normal edge.
    This gives us the clean separation: search and generation are decoupled.
    """
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
    Final graph structure:

      START → retrieve → grade → generate   → END  (documents path)
                              ↘ web_search → generate → END  (web path)

    generate sits at the end of BOTH paths.
    web_search sits only on the fallback path.
    """
    graph = StateGraph(GraphState)

    graph.add_node("retrieve",   retrieve)
    graph.add_node("grade",      grade)
    graph.add_node("web_search", web_search)
    graph.add_node("generate",   generate)

    graph.add_edge(START,        "retrieve")
    graph.add_edge("retrieve",   "grade")

    graph.add_conditional_edges(
        "grade",
        should_search_web,
        {
            "generate":   "generate",
            "web_search": "web_search",
        },
    )

    graph.add_edge("web_search", "generate")  # web_search always flows to generate
    graph.add_edge("generate",   END)

    compiled = graph.compile()
    print("✓ Graph compiled successfully")
    return compiled


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def run_query(question: str):
    """Run a single question through the full RAG graph."""

    print("=" * 55)
    print("  ResearchMind — Phase 3: Multi-Agent RAG")
    print("=" * 55)
    print(f"\nQuestion: {question}\n")

    graph = build_graph()

    result = graph.invoke({
        "question":       question,
        "chunks":         [],
        "grade":          "",
        "search_results": [],
        "source_type":    "documents",
        "answer":         "",
        "sources":        [],
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
    # Test 1: should be answered from documents
    run_query("What is retrieval-augmented generation?")

    print("\n\n")

    # Test 2: not in documents — should trigger web search fallback
    run_query("What is the capital of France?")
"""
Agent wrapper for LocalAIAgentWithRAG.

Behavior:
  1) Try to build a ReAct agent using langchain_community.agent_toolkits.create_react_agent.
  2) If that fails (API mismatch), use a simple fallback orchestrator:
       - First call the RAG tool (agnikul_rag_search)
       - If RAG returns nothing, call the Wikipedia tool (wikipedia_search)
  3) Expose run_agent(question: str) -> str to be used by main.py.
"""

from typing import Optional
import traceback

# Core model & tools imports (should exist)
from langchain_ollama.llms import OllamaLLM

# Our tool registry (Tool objects)
from tools.registry import TOOLS

# Also import direct callable wrappers so fallback can call them
try:
    # Your registry exposes Tool objects; import underlying callables directly
    from tools.rag_tool import rag_search
except Exception:
    rag_search = None

try:
    from tools.wiki_tool import wiki_search
except Exception:
    wiki_search = None

# Attempt to import community agent toolkit (may or may not be available)
_create_agent_available = False
try:
    # Preferred API
    from langchain_community.agent_toolkits import create_react_agent
    from langchain.agents import AgentExecutor  # sometimes provided by experimental package
    _create_agent_available = True
except Exception:
    _create_agent_available = False

# Initialize model (shared)
_model = OllamaLLM(model="llama3.1:8b", base_url="http://localhost:11434")

# If agent toolkit is available, build agent now (lazy-safe)
_agent_executor: Optional[object] = None
if _create_agent_available:
    try:
        # create_react_agent typically accepts (llm, tools, verbose=..)
        # different versions may vary; wrap in try so failures fall back
        _agent_executor = create_react_agent(llm=_model, tools=TOOLS, verbose=False)
    except Exception:
        # If it fails, we will fallback at runtime
        _agent_executor = None

def _fallback_orchestrator(question: str) -> str:
    """
    Deterministic fallback:
      1) Call RAG; if results found and appear relevant, return them.
      2) Else call Wikipedia and return that.
      3) Else say nothing found.

    Relevance test: we consider RAG "not found" when the rag_search result
    contains phrases like 'no relevant' or 'No relevant documents' or is very short.
    """
    try:
        # 1) Try RAG first (if available)
        if rag_search:
            rag_res = rag_search(question)
            if rag_res:
                low = rag_res.lower()
                # treat 'no relevant' responses as empty
                if ("no relevant" not in low) and ("no relevant documents" not in low):
                    # Heuristic: if rag_res contains the query term or "topic:" we treat as relevant.
                    q_tokens = [t.lower() for t in question.split() if len(t) > 2]
                    matched = any(tok in low for tok in q_tokens[:5])  # check first few tokens
                    # also accept if result length is reasonably long
                    if matched or len(rag_res.split()) > 20:
                        return f"[RAG]\n\n{rag_res}"
            # fallthrough if rag_res empty / not relevant
        # 2) RAG empty or not relevant: call Wikipedia (if available)
        if wiki_search:
            wiki_res = wiki_search(question)
            if wiki_res:
                return f"[Wikipedia]\n\n{wiki_res}"
    except Exception as e:
        import traceback
        return f"Fallback orchestrator error: {e}\n\nTraceback:\n{traceback.format_exc()}"

    return "No information found in local RAG or Wikipedia."

def run_agent(question: str) -> str:
    """
    Main entrypoint used by main.py.
    It will try to run the full agent if available; otherwise use the fallback orchestrator.
    Returns the final string response.
    """
    # First, if we have an initialized agent executor, try it
    global _agent_executor
    if _agent_executor:
        try:
            # many agent executors expose a .run() or .invoke() method
            if hasattr(_agent_executor, "run"):
                return _agent_executor.run(question)
            if hasattr(_agent_executor, "invoke"):
                return _agent_executor.invoke(question)
            # last resort: call as callable
            return _agent_executor(question)
        except Exception:
            # If agent fails, fall back (but surface the error for debugging)
            tb = traceback.format_exc()
            fallback = _fallback_orchestrator(question)
            return f"Agent execution failed, falling back.\n\nAgent error:\n{tb}\n\nFallback result:\n{fallback}"

    # If no agent executor, use fallback orchestrator
    return _fallback_orchestrator(question)


# Optional: small CLI test when running this file directly
if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not q:
        q = input("Question: ").strip()
    print(run_agent(q))

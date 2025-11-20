# tools/rag_tool.py
from langchain_core.tools import Tool
import traceback

# retrieval
from vector import retriever
try:
    from vector import vector_store
except Exception:
    vector_store = None

# local LLM (used to summarize retrieved docs)
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# instantiate a short-lived local model for summarization
_summarizer_llm = OllamaLLM(model="gemma3:latest", base_url="http://localhost")
_summary_template = """You are a concise, factual assistant. Use ONLY the CONTEXT to answer the QUESTION below.
If the context does not contain the answer, say "I don't know (not in context)."

### CONTEXT:
{context}

### QUESTION:
{question}

### INSTRUCTIONS:
- Answer in 2-4 short sentences.
- At the end, add a "Sources:" line with the distinct source names from the context.
"""

_prompt = PromptTemplate.from_template(_summary_template)
_summary_chain = _prompt | _summarizer_llm | StrOutputParser()



def _call_retriever(query: str, k: int = 5):
    candidates = [
        "get_relevant_documents",
        "retrieve",
        "_get_relevant_documents",
        "_retrieve",
        "get_relevant_results",
        "similarity_search"
    ]

    for name in candidates:
        fn = getattr(retriever, name, None)
        if callable(fn):
            try:
                docs = fn(query)
                return docs
            except TypeError:
                try:
                    docs = fn(query, search_kwargs={"k": k})
                    return docs
                except Exception:
                    continue
            except Exception:
                continue

    # fallback to vector_store methods
    if vector_store is not None:
        try:
            if hasattr(vector_store, "similarity_search"):
                return vector_store.similarity_search(query, k=k)
            if hasattr(vector_store, "search"):
                return vector_store.search(query, k=k)
        except Exception:
            return []
    return []


def rag_search(query: str) -> str:
    """
    Retrieve top docs, then ask the LLM to synthesize an answer (with sources).
    Returns a short, LLM-generated summary + Sources line.
    """
    try:
        docs = _call_retriever(query, k=5)
    except Exception as e:
        return f"RAG retriever error: {e}\n\nTraceback:\n{traceback.format_exc()}"

    if not docs:
        return "No relevant documents found in the local dataset."

    # Build a concise context and collect sources
    snippets = []
    sources = []
    for d in docs:
        text = getattr(d, "page_content", None) or getattr(d, "content", None) or str(d)
        snippets.append(text)
        meta = getattr(d, "metadata", None) or {}
        if isinstance(meta, dict):
            src = meta.get("source")
            if src:
                sources.append(src)

    context_text = "\n\n".join(snippets)
    # distill unique, short source names
    unique_sources = ", ".join(sorted(set(sources))) if sources else "no source metadata"

    # Run the summarization chain (safe: truncate context if huge)
    MAX_CHARS = 12000
    if len(context_text) > MAX_CHARS:
        context_text = context_text[:MAX_CHARS] + "\n\n[TRUNCATED]"

    try:
        result = _summary_chain.invoke({"context": context_text, "question": query})
    except Exception as e:
        # if summarization fails, fall back to returning short combined passages with provenance
        fallback = "\n\n".join(snippets[:3])
        return f"Summarization error: {e}\n\nFallback passages:\n\n{fallback}\n\nSources: {unique_sources}"

    # Guarantee a Sources line exists
    if "Sources:" not in result:
        result = f"{result}\n\nSources: {unique_sources}"

    return result


# expose Tool for agent registries
rag_tool = Tool(
    name="agnikul_rag_search",
    func=rag_search,
    description="Search the internal Agnikul dataset, summarize findings, and return a short answer with sources."
)


from langchain_community.utilities.arxiv import ArxivAPIWrapper

# Create a singleton wrapper like wiki_tool
# Configure sensible defaults; adjust top_k_results or doc_content_chars_max as needed.
_arxiv_wrapper = ArxivAPIWrapper(
    top_k_results=3,          # how many top results to summarize
    load_max_docs=3,          # maximum docs to fetch/load
    doc_content_chars_max=4000  # truncate long contents
)

def arxiv_search(query: str) -> str:
    """
    Run an ArXiv lookup and return a cleaned string.
    Uses langchain_community.utilities.arxiv.ArxivAPIWrapper.run under the hood.
    Returns a short summary of top results or an error message.
    """
    try:
        if not query or not isinstance(query, str):
            return "ERROR: Expected a non-empty query string for arXiv search."
        # ArxivAPIWrapper.run returns a single string with summaries for the top-k results.
        result = _arxiv_wrapper.run(query)
        if not result:
            return "No arXiv results found."
        return result
    except Exception as e:
        return f"ArXiv lookup error: {e}"

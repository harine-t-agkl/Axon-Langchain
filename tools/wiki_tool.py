# tools/wiki_tool.py
from langchain_community.utilities import WikipediaAPIWrapper

# Create a singleton wrapper so we don't recreate it on every call
_wiki_wrapper = WikipediaAPIWrapper(
    top_k_results=2,
    doc_content_chars_max=4000
)

def wiki_search(query: str) -> str:
    """
    Run a Wikipedia lookup and return a cleaned string.
    Uses langchain_community.utilities.WikipediaAPIWrapper under the hood.
    Returns a short summary or an error message.
    """
    try:
        # WikipediaAPIWrapper exposes a run() method that returns text
        result = _wiki_wrapper.run(query)
        if not result:
            return "No Wikipedia results found."
        return result
    except Exception as e:
        return f"Wikipedia lookup error: {e}"

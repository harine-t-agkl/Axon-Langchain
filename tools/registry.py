from langchain_core.tools import Tool
from tools.wiki_tool import wiki_search
from tools.rag_tool import rag_search
from tools.duckduckgo_tool import ddg_search
from tools.bibtex_tool import parse_bibtex_tool
from tools.arxiv_tool import arxiv_search 

TOOLS = [
    Tool(
        name="wikipedia_search",
        func=wiki_search,
        description="Search Wikipedia and return a summary and source URL."
    ),
    Tool(
        name="agnikul_rag_search",
        func=rag_search,
        description="Search internal Agnikul dataset for relevant passages and return them with sources."
    ),
    Tool(
        name="duckduckgo_search",
        func=ddg_search,
        description="Use this tool to search the web using DuckDuckGo. Best for real-time information."
    ),
    Tool(
        name="bibtex",
        func=parse_bibtex_tool,
        description="Parse a .bib file or raw BibTeX text and return entries as JSON (input: path or raw content)."
    ),
    Tool(
        name="arxiv_search",                  
        func=arxiv_search,
        description="Search arXiv for academic papers and return summaries of top results."
    )
]

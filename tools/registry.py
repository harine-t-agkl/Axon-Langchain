from langchain_core.tools import Tool
from tools.wiki_tool import wiki_search
from tools.rag_tool import rag_search

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
    )
]

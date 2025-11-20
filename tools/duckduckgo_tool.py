from ddgs import DDGS

def ddg_search(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
            formatted = []
            for r in results:
                title = r.get("title")
                href = r.get("href")
                snippet = r.get("body") or ""
                formatted.append(f"{title}\n{href}\n{snippet}\n")
            return "\n".join(formatted) if formatted else "No results."
    except Exception as e:
        return f"DDG error: {str(e)}"

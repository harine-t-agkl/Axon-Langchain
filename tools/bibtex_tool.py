
from langchain_community.utilities.bibtex import BibtexparserWrapper
import tempfile
from pathlib import Path
import json

# Create a singleton wrapper 
_bibtex_wrapper = BibtexparserWrapper()


def _write_temp_bib(content: str) -> str:
    """
    If the input is raw BibTeX text, write it to a temporary .bib file
    and return the temp file path.
    """
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".bib", mode="w", encoding="utf-8")
    tf.write(content)
    tf.flush()
    tf.close()
    return tf.name


def parse_bibtex_tool(input_text: str) -> str:
    """
    Parse a .bib file OR raw BibTeX content.

    Accepts:
    - Path to a .bib file
    - Raw BibTeX starting with '@'

    Returns:
    - JSON string of parsed entries
    - Or an error message
    """
    if not input_text or not isinstance(input_text, str):
        return "ERROR: Expected path to a .bib file or raw BibTeX content."

    input_text = input_text.strip()

    # Case 1: Raw BibTeX string
    if input_text.startswith("@"):
        try:
            input_text = _write_temp_bib(input_text)
        except Exception as e:
            return f"ERROR writing temporary BibTeX file: {e}"

    # Case 2: Path to .bib file
    path = Path(input_text)
    if not path.exists():
        return f"ERROR: BibTeX file not found: {input_text}"

    try:
        entries = _bibtex_wrapper.load_bibtex_entries(str(path))
        return json.dumps(entries, indent=2, ensure_ascii=False)

    except Exception as e:
        return f"BibTeX parsing error: {e}"

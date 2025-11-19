# vector.py
import os
import pandas as pd

# LangChain-style imports
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# Load dataset (expects dataset.csv in repo root)
df = pd.read_csv("dataset.csv", quotechar='"', escapechar='\\')

# Embedding model (Ollama embeddings running locally)
embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://localhost:11434"
)

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

# Prepare documents list if first run
documents = []
ids = []

if add_documents:
    for i, row in df.iterrows():
        # combine all columns as a single text block
        combined_text = (
            f"Topic: {row.get('Topic','')}\n"
            f"Subtopic: {row.get('Subtopic','')}\n"
            f"Detail: {row.get('Detail','')}\n"
            f"Date: {row.get('Date','')}\n"
            f"Source: {row.get('Source','')}"
        )

        doc = Document(
            page_content=combined_text,
            metadata={
                "topic": row.get("Topic", ""),
                "subtopic": row.get("Subtopic", ""),
                "date": row.get("Date", ""),
                "source": row.get("Source", "")
            }
        )
        documents.append(doc)
        ids.append(str(i))

# Create / Load Chroma DB
# Note: Chroma constructor parameters may vary by langchain version;
# this pattern should work for modern LangChain where Chroma accepts persist_directory & embedding_function.
vector_store = Chroma(
    persist_directory=db_location,
    embedding_function=embeddings,
    collection_name="agnikul_data"
)

# Only add documents on first run
if add_documents and documents:
    vector_store.add_documents(documents=documents, ids=ids)
    # ensure persistence
    try:
        vector_store.persist()
    except Exception:
        # some Chroma wrappers auto-persist; ignore if not available
        pass

# Retriever with top-k results
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

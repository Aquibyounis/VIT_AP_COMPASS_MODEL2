# embed.py
"""
Create embeddings for your dataset and store them in Chroma DB.
Also provides helper functions to use embeddings in eval.py.
"""

import json
import os
import numpy as np
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

# -------------------------------
# CONFIG
# -------------------------------
DATA_FILE = "procedures.json"        # your structured dataset
DB_DIR = "chroma_db"                 # persistent database folder
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # good general-purpose model
CHUNK_SIZE = 350                     # characters per chunk
CHUNK_OVERLAP = 50                   # overlap to preserve context

# -------------------------------
# EMBEDDING MODEL (GLOBAL)
# -------------------------------
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def load_dataset(path):
    """Load dataset from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split long text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def build_documents(dataset):
    """Convert JSON dataset into LangChain Document objects with metadata."""
    docs = []
    for item in dataset:
        title = item["title"]
        content = item["content"]

        # chunk if necessary
        chunks = chunk_text(content) if len(content) > CHUNK_SIZE else [content]

        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=f"{title}\n\n{chunk}",
                    metadata={"title": title, "chunk": i}
                )
            )
    return docs


# -------------------------------
# EXTRA FUNCTIONS (for eval.py)
# -------------------------------
def get_embedding(text: str):
    """Return embedding vector for input text."""
    return embeddings.embed_query(text)


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


# -------------------------------
# MAIN
# -------------------------------
def main():
    # remove old DB if exists
    if os.path.exists(DB_DIR):
        print(f"🗑️ Removing existing DB at {DB_DIR}...")
        import shutil
        shutil.rmtree(DB_DIR)

    print("📂 Loading dataset...")
    dataset = load_dataset(DATA_FILE)

    print("📝 Building documents...")
    documents = build_documents(dataset)

    print("💾 Creating Chroma DB...")
    db = Chroma.from_documents(documents, embedding=embeddings, persist_directory=DB_DIR)

    print("✅ Embeddings stored successfully in", DB_DIR)


if __name__ == "__main__":
    main()

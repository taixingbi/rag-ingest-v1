import os
import sys
import unicodedata
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

# Set ENV and CHROMA_DATABASE from CLI when run as script (CLI wins over .env)
if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("dev", "qa", "prod"):
        sys.exit("Usage: python ingest.py {dev|qa|prod} [save <database_name>]")
    _env = sys.argv[1]
    os.environ["ENV"] = _env
    load_dotenv()
    load_dotenv(f".env.{_env}")
    # Always set database from env so "dev" → rag_dev (avoids .env typo e.g. CHROMA_DATABASE=rag)
    if len(sys.argv) >= 4 and sys.argv[2] == "save":
        os.environ["CHROMA_DATABASE"] = sys.argv[3]
    else:
        os.environ["CHROMA_DATABASE"] = f"rag_{_env}"
else:
    load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import (
    PROJECT_ROOT,
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
    EMBEDDING_MODEL,
    CHROMA_SETTINGS,
    get_chroma_client,
)


# ----------------------------
# Read
# ----------------------------
def read_utf8(path: Path | str) -> str:
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        return f.read()


# ----------------------------
# Normalize
# ----------------------------
def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


# ----------------------------
# Chunk
# ----------------------------
@lru_cache(maxsize=1)
def build_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        add_start_index=True,
    )


def chunk_text(text: str, source: str) -> list:
    splitter = build_splitter()
    docs = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source, "category": "tb"}],
    )

    created_at = datetime.now(timezone.utc).isoformat()

    for i, d in enumerate(docs):
        start = d.metadata.get("start_index", 0)
        d.metadata.update({
            "chunk_index": i,
            "chunk_id": f"{source}::chunk_{i:04d}",
            "char_start": start,
            "char_end": start + len(d.page_content),
            "created_at": created_at,
        })

    return docs


# ----------------------------
# Embed + Save (Chroma Cloud)
# ----------------------------
def upsert_chroma(docs: list):
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    chroma_client = get_chroma_client()

    vectordb = Chroma(
        client=chroma_client,
        collection_name=CHROMA_SETTINGS["collection_name"],
        embedding_function=embeddings,
    )

    vectordb.add_documents(docs)
    return vectordb


# ----------------------------
# Corpus loading
# ----------------------------
def load_corpus() -> list:
    """Load and chunk all data files. Returns list of Documents with chunk_id metadata."""
    if not DATA_DIR.exists():
        return []
    all_docs = []
    for txt_file in sorted(DATA_DIR.glob("*.txt")):
        raw = read_utf8(txt_file)
        clean = normalize_unicode(raw)
        docs = chunk_text(clean, source=str(txt_file.relative_to(PROJECT_ROOT)))
        all_docs.extend(docs)
    return all_docs


# ----------------------------
# Pipeline entry
# ----------------------------
def run_ingest() -> None:
    env = os.getenv("ENV", "dev")
    print(f"📄 Environment: {env}")
    print("📄 Loading and chunking documents...")
    all_docs = load_corpus()
    if not all_docs:
        print("⚠️ No documents found in data/ — nothing to ingest.")
        return
    print(f"🧠 Embedding + saving ({len(all_docs)} chunks) to Chroma Cloud")
    upsert_chroma(all_docs)
    print("✅ Ingest finished")


if __name__ == "__main__":
    run_ingest()

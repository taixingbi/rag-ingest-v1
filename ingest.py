import json
import os
import unicodedata
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from cli import setup_cli_env

if __name__ == "__main__":
    setup_cli_env()
else:
    load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    DATA_DIR,
    FILE_TO_COLLECTION,
    PROJECT_ROOT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
)
from chroma_ingest import upsert_chroma


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
# Corpus loading (JSON)
# ----------------------------
FILES_JSON = DATA_DIR / "files.json"


def _items_from_data(data: list | dict) -> list:
    """Normalize JSON root to a list of items."""
    if isinstance(data, list):
        return data
    items = data.get("files", data.get("documents", [data]))
    return items if isinstance(items, list) else [items]


def _text_from_item(item: str | dict, index: int, base_source: str) -> tuple[str, str]:
    """Extract (text, source) from a JSON item."""
    default_source = f"{base_source}::{index}"
    if isinstance(item, str):
        return item, default_source
    if isinstance(item, dict):
        text = item.get("text") or item.get("content") or item.get("body")
        if text is None and ("q" in item or "a" in item):
            text = " ".join(str(item.get(k, "")) for k in ("q", "a") if item.get(k))
        if text is None:
            text = json.dumps(item, ensure_ascii=False)
        source = item.get("id") or item.get("source") or default_source
        return str(text), str(source)
    return str(item), default_source


def _docs_from_items(items: list, base_source: str) -> list:
    """Chunk a list of JSON items into Documents."""
    docs = []
    for i, item in enumerate(items):
        text, source = _text_from_item(item, i, base_source)
        docs.extend(chunk_text(normalize_unicode(text), source=source))
    return docs


def load_corpus() -> list:
    """Load from data/files.json or all data/*.json; chunk and return Documents (single collection)."""
    if FILES_JSON.exists():
        data = json.loads(read_utf8(FILES_JSON))
        return _docs_from_items(_items_from_data(data), "data/files.json")
    if not DATA_DIR.exists():
        return []
    all_docs = []
    for path in sorted(DATA_DIR.glob("*.json")):
        data = json.loads(read_utf8(path))
        base = str(path.relative_to(PROJECT_ROOT))
        all_docs.extend(_docs_from_items(_items_from_data(data), base))
    return all_docs


def load_corpus_by_collection() -> list[tuple[str, list]]:
    """Load data/*.json; for each file mapped in FILE_TO_COLLECTION, yield (collection_name, docs)."""
    if not DATA_DIR.exists():
        return []
    out = []
    for path in sorted(DATA_DIR.glob("*.json")):
        collection = FILE_TO_COLLECTION.get(path.name)
        if not collection:
            continue
        data = json.loads(read_utf8(path))
        base = str(path.relative_to(PROJECT_ROOT))
        docs = _docs_from_items(_items_from_data(data), base)
        if docs:
            out.append((collection, docs))
    return out


# ----------------------------
# Pipeline entry
# ----------------------------
def run_ingest() -> None:
    env = os.getenv("ENV", "dev")
    print(f"📄 Environment: {env}")
    # Ingest by collection: profile.json → taixing_identity, resume.json → taixing_resume, qa.json → taixing_qa
    by_collection = load_corpus_by_collection()
    if not by_collection:
        all_docs = load_corpus()
        if not all_docs:
            print("⚠️ No documents found in data/files.json or data/*.json — nothing to ingest.")
            return
        print(f"🧠 Embedding + saving ({len(all_docs)} chunks) to Chroma Cloud")
        upsert_chroma(all_docs)
    else:
        for collection_name, docs in by_collection:
            print(f"🧠 {collection_name}: {len(docs)} chunks → Chroma Cloud")
            upsert_chroma(docs, collection_name=collection_name)
    print("✅ Ingest finished")


if __name__ == "__main__":
    run_ingest()

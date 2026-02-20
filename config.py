"""
Configuration loaded from environment (.env).
"""

from dataclasses import dataclass
import os

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    mongodb_uri: str = os.environ.get("MONGODB_URI", "")
    mongodb_db: str = os.environ.get("MONGODB_DB", "rag")
    mongodb_collection: str = os.environ.get("MONGODB_COLLECTION", "rag_chunks")

    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    embed_model: str = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    # Chunking (token-based preferred, char-based fallback)
    chunk_tokens: int = int(os.environ.get("CHUNK_TOKENS", "1000"))
    overlap_tokens: int = int(os.environ.get("OVERLAP_TOKENS", "150"))
    chunk_chars: int = int(os.environ.get("CHUNK_CHARS", "5000"))
    overlap_chars: int = int(os.environ.get("OVERLAP_CHARS", "800"))

    # Ingest (batch size for embeddings API)
    batch_size: int = int(os.environ.get("BATCH_SIZE", "64"))

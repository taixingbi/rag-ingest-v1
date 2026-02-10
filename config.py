import os
from pathlib import Path

# ======================
# Paths
# ======================
ENV = os.getenv("ENV", "dev")
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

# ======================
# Chunking
# ======================
CHUNK_SIZE = 2200  # chars ≈ 800–1200 tokens
CHUNK_OVERLAP = 300
SEPARATORS = ("\n\n", "\n", "。", ".", "！", "!", "？", "?", ";", "；", ",", "，", " ", "")

# ======================
# Embedding
# ======================
EMBEDDING_MODEL = "text-embedding-3-small"

# ======================
# Chroma (cloud) — from .env: CHROMA_API_KEY_{ENV}, CHROMA_TENANT_{ENV}, CHROMA_DATABASE_{ENV}
# ======================
_env_upper = ENV.upper()
CHROMA_API_KEY = os.getenv("CHROMA_API_KEY") or os.getenv(f"CHROMA_API_KEY_{_env_upper}")
CHROMA_TENANT = os.getenv("CHROMA_TENANT") or os.getenv(f"CHROMA_TENANT_{_env_upper}")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE") or os.getenv(f"CHROMA_DATABASE_{_env_upper}") or f"rag_{ENV}"
CHROMA_SETTINGS = {
    "api_key": CHROMA_API_KEY,
    "tenant": CHROMA_TENANT,
    "database": CHROMA_DATABASE,
    "collection_name": "tb_all",
}


def get_chroma_client():
    """Create Chroma Cloud client from config."""
    import chromadb
    return chromadb.CloudClient(
        api_key=CHROMA_API_KEY,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE,
    )

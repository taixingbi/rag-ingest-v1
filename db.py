from typing import Any, Dict, List
from pymongo.operations import UpdateOne


def ensure_unique_index(col) -> None:
    """
    Ensure indexes exist for efficient queries.
    Note: _id is already unique and indexed by default in MongoDB.
    """
    # Index chunk_id for queries (unique to prevent duplicates)
    try:
        col.create_index("chunk_id", unique=True)
    except Exception:
        # Index might already exist, that's fine
        pass
    
    # Index source.source_id for filtering
    try:
        col.create_index("source.source_id")
    except Exception:
        pass
    
    # Index metadata.tags for filtering
    try:
        col.create_index("metadata.tags")
    except Exception:
        pass


def upsert_chunks(col, docs: List[Dict[str, Any]]) -> None:
    """
    Bulk upsert chunks using stable _id.
    Uses UpdateOne with upsert=True for idempotent ingestion.
    """
    ops = []
    for d in docs:
        doc_id = d["_id"]
        ops.append(
            UpdateOne(
                {"_id": doc_id},
                {"$set": d},
                upsert=True,
            )
        )
    if ops:
        col.bulk_write(ops, ordered=False)

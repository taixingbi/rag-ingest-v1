# RAG Pipeline

Hybrid search (BM25 + dense) over documents with Chroma Cloud and LangChain.

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```
OPENAI_API_KEY=your-openai-key
CHROMA_API_KEY=your-chroma-cloud-api-key
# Optional: CHROMA_TENANT=... CHROMA_DATABASE=...
```

## Usage

**Ingest** — index `data/*.txt` into Chroma Cloud and build BM25 index:

```bash
python ingest.py dev   # dev → rag_dev
python ingest.py qa    # qa → rag_qa
python ingest.py prod  # prod → rag_prod
```

**Query** — RAG over indexed documents:

```bash
python query.py "your question"
# or: python test_query.py "your question"
# interactive: python query.py
```

## MCP server

Expose RAG as MCP tools (for Cursor, Claude Desktop, etc.):

```bash
# stdio (default, for Cursor)
python mcp_server.py

# SSE / Streamable HTTP (requires fastapi, uvicorn)
python mcp_server.py --transport streamable-http --port 8000
# Connect to http://localhost:8000/mcp
```

## GitHub Actions

On push, ingest runs by branch (see `.github/workflows/ingest.yml`):

| Branch        | Command              |
|---------------|----------------------|
| `feature/**`  | `python ingest.py dev`  |
| `qa`          | `python ingest.py qa`   |
| `main`        | `python ingest.py prod` |

Add repo secrets: `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_TENANT`.

## Pipeline

1. **Ingest** — Load docs from `data/`, normalize, chunk, embed; store in Chroma Cloud; build BM25 index (`rank_bm25`) under `vector_store/bm25/`
2. **Retrieve** — Hybrid: BM25 candidates → dense rerank (Chroma) → combine scores
3. **Generate** — Prompt LLM with retrieved context

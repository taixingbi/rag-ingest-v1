# RAG Ingest

Index `data/*.txt` into Chroma Cloud (dense embeddings only). Collection name: `tb_all`.

[Chroma dashboard](https://www.trychroma.com/hunter)

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env` with per-env Chroma vars: `CHROMA_API_KEY_DEV`, `CHROMA_TENANT_DEV`, `CHROMA_DATABASE_DEV` (and `_QA`, `_PROD` for other envs). Set `OPENAI_API_KEY` for embeddings.

## Usage

**Ingest** — load, chunk, embed, and send to Chroma Cloud:

```bash
python ingest.py dev    # → rag_dev
python ingest.py qa     # → rag_qa
python ingest.py prod   # → rag_prod
```

Optional: `python ingest.py dev save <database_name>` to override the database.

## GitHub Actions

On push, ingest runs by branch (see `.github/workflows/ingest.yml` if present):

| Branch       | Command               |
|-------------|------------------------|
| `feature/**`| `python ingest.py dev` |
| `qa`        | `python ingest.py qa`  |
| `main`      | `python ingest.py prod`|

Repo secrets: `OPENAI_API_KEY`, `CHROMA_API_KEY`, `CHROMA_TENANT`.

## Pipeline

1. **Ingest** — Read `data/*.txt`, normalize, chunk, embed (OpenAI), upsert to Chroma Cloud collection `tb_all`.




### resume
Convert Resume Into Structured Knowledge as json with download link
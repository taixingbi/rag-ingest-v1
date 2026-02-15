"""Upsert documents to Chroma Cloud."""
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from config import EMBEDDING_MODEL, CHROMA_SETTINGS, get_chroma_client


def upsert_chroma(docs: list, collection_name: str | None = None) -> None:
    """Embed docs and add them to the given Chroma Cloud collection."""
    name = collection_name or CHROMA_SETTINGS["collection_name"]
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    chroma_client = get_chroma_client()
    vectordb = Chroma(
        client=chroma_client,
        collection_name=name,
        embedding_function=embeddings,
    )
    vectordb.add_documents(docs)

import chromadb

from admissions_agent.config import settings


def get_client():
    return chromadb.PersistentClient(path=str(settings.chroma_dir))


def get_collection():
    # No embedding function is passed, so ChromaDB uses its default local model (all-MiniLM-L6-v2).
    return get_client().get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection():
    client = get_client()
    if settings.collection_name in [collection.name for collection in client.list_collections()]:
        client.delete_collection(settings.collection_name)
    return get_collection()

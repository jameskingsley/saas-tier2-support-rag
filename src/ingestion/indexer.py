import json
import os
import sys
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient, models

# Ensure project root is in sys.path when running directly
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import settings


def get_qdrant_client() -> QdrantClient:
    """Initializes Qdrant client (Cloud if QDRANT_URL is set, otherwise local disk)."""
    if settings.qdrant_url:
        print(f"[INFO] Connecting to Qdrant Cloud: {settings.qdrant_url}")
        return QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=60,
        )
    print(f"[INFO] Using local Qdrant storage at: {settings.qdrant_path}")
    os.makedirs(settings.qdrant_path, exist_ok=True)
    return QdrantClient(path=settings.qdrant_path)


def setup_collection(client: QdrantClient, collection_name: str):
    """Creates the hybrid collection and payload indexes if they do not exist."""
    if client.collection_exists(collection_name):
        print(f"[INFO] Collection '{collection_name}' already exists.")
        return

    print(f"[INFO] Creating hybrid collection '{collection_name}'...")
    dense_size = client.get_embedding_size(settings.dense_model)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_size,
                distance=models.Distance.COSINE,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF,
            )
        },
    )

    # Create payload indexes upfront for Qdrant Cloud strict-mode filtering
    for field in ["source_type", "category", "priority"]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
    print(f"[SUCCESS] Created collection '{collection_name}' with dense ({dense_size}d) + sparse BM25 vectors and payload indexes.")


def index_knowledge_base(cards_path: str = "data/processed/kb_cards.jsonl", batch_size: int = 50):
    """Reads processed Root-Cause Cards and upserts them into Qdrant in batches."""
    if not os.path.exists(cards_path):
        raise FileNotFoundError(f"Processed cards not found at {cards_path}. Run chunker.py first.")

    with open(cards_path, "r", encoding="utf-8") as f:
        cards = [json.loads(line) for line in f if line.strip()]

    print(f"[INFO] Loaded {len(cards)} Root-Cause Cards from {cards_path}")

    client = get_qdrant_client()
    setup_collection(client, settings.collection_name)

    print(f"[INFO] Computing embeddings ({settings.dense_model} + {settings.sparse_model}) and uploading...")
    for i in tqdm(range(0, len(cards), batch_size), desc="Indexing batches"):
        batch = cards[i : i + batch_size]
        points = [
            models.PointStruct(
                id=card["id"],
                vector={
                    "dense": models.Document(text=card["text"], model=settings.dense_model),
                    "sparse": models.Document(text=card["text"], model=settings.sparse_model),
                },
                payload={
                    "title": card["title"],
                    "source_type": card["source_type"],
                    "category": card["category"],
                    "priority": card["priority"],
                    "text": card["text"],
                    "metadata": card.get("metadata", {}),
                },
            )
            for card in batch
        ]
        client.upsert(
            collection_name=settings.collection_name,
            points=points,
            wait=True,
        )

    info = client.get_collection(settings.collection_name)
    print(f"[SUCCESS] Indexing complete! Total points in '{settings.collection_name}': {info.points_count}")


if __name__ == "__main__":
    index_knowledge_base()
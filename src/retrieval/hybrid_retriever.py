import sys
from pathlib import Path
from qdrant_client import QdrantClient, models
from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

# Ensure project root is in sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.config import settings


class HybridRetriever:
    def __init__(self):
        print("[INFO] Initializing Hybrid Retriever and connecting to Qdrant...")
        if settings.qdrant_url:
            self.client = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key,
                timeout=60,
            )
        else:
            self.client = QdrantClient(path=settings.qdrant_path)

        # Initialize local FastEmbed models for query encoding
        self.dense_model = TextEmbedding(model_name=settings.dense_model)
        self.sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)

    def search(self, query: str, limit: int = 5, score_threshold: float = 0.0) -> list[dict]:
        """Performs hybrid dense + sparse retrieval with Qdrant query API."""
        # Generate query vectors
        dense_vector = list(self.dense_model.embed([query]))[0].tolist()
        sparse_tokens = list(self.sparse_model.embed([query]))[0]
        sparse_vector = models.SparseVector(
            indices=sparse_tokens.indices.tolist(),
            values=sparse_tokens.values.tolist()
        )

        # Perform hybrid search using prefetch
        results = self.client.query_points(
            collection_name=settings.collection_name,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using="dense",
                    limit=limit * 2,
                ),
                models.Prefetch(
                    query=sparse_vector,
                    using="sparse",
                    limit=limit * 2,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
        )

        retrieved_cards = []
        for point in results.points:
            payload = point.payload or {}
            retrieved_cards.append({
                "id": str(point.id),
                "score": float(point.score),
                "title": payload.get("title", ""),
                "source_type": payload.get("source_type", ""),
                "category": payload.get("category", ""),
                "priority": payload.get("priority", ""),
                "text": payload.get("text", ""),
                "metadata": payload.get("metadata", {}),
            })

        return retrieved_cards


if __name__ == "__main__":
    retriever = HybridRetriever()
    test_query = "Webhook delivery timeout error 504"
    print(f"\n[TEST] Running test query: '{test_query}'")
    hits = retriever.search(test_query, limit=3)
    for idx, hit in enumerate(hits, 1):
        print(f"\n--- Hit {idx} (Score: {hit['score']:.4f}) ---")
        print(f"Title: {hit['title']}")
        print(f"Type: {hit['source_type']} | Category: {hit['category']}")
        print(f"Snippet: {hit['text'][:200]}...")
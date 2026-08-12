"""Thin wrapper around a local sentence-transformers model.

We use a local, free embedding model (all-MiniLM-L6-v2, 384 dims) instead of a paid
embedding API. This keeps ingestion cheap/offline and avoids a second API key, at the
cost of slightly lower embedding quality than e.g. OpenAI's text-embedding-3-small.
For a 4-6 doc / ~50 chunk corpus this tradeoff is a non-issue in practice.
"""

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]

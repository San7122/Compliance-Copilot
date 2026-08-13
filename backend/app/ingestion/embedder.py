"""Embedding service backed by a hosted API.

Why hosted rather than local: the previous implementation used
`sentence-transformers`, which pulls in torch and made the backend image ~3.3GB. That
is a large amount of infrastructure to carry for a corpus of ~1,000 short clauses, and
it dominated build time and disk. A hosted embedding endpoint removes the entire ML
stack from the image.

Why `text-embedding-3-small` specifically: it supports a `dimensions` request
parameter, so its output can be reduced to the 384 dimensions the existing schema
already uses. That means switching providers needs no `vector(N)` migration and no
change to the pgvector column. The dimension is asserted on every response rather than
assumed -- if the API ever returns a different width, ingestion fails loudly instead of
writing vectors that silently mismatch the column.

The single most important property here is that documents and queries are embedded by
the same model with the same configuration. Mixing models puts the two sets of vectors
in different spaces, and cosine similarity between them becomes meaningless -- retrieval
degrades quietly rather than erroring, which makes it a nasty bug to find. Both paths
below therefore route through one `_embed` call.
"""

from functools import lru_cache

from openai import OpenAI

from app.config import settings

# The API accepts many inputs per request; batching keeps request bodies sane while
# still avoiding one round trip per chunk.
_BATCH_SIZE = 128


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    if not settings.embedding_api_key:
        raise RuntimeError(
            "No embedding API key configured. Set EMBEDDING_API_KEY (or OPENAI_API_KEY) "
            "in .env -- ingestion and retrieval both need it."
        )
    return OpenAI(api_key=settings.embedding_api_key)


def _embed(texts: list[str]) -> list[list[float]]:
    """The one place embeddings are produced, for documents and queries alike."""
    if not texts:
        return []

    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[start : start + _BATCH_SIZE]
        response = _client().embeddings.create(
            model=settings.embedding_model,
            input=batch,
            dimensions=settings.embedding_dimensions,
        )
        # The API returns items in request order, but it also carries an explicit index;
        # sorting by it removes the assumption entirely.
        for item in sorted(response.data, key=lambda d: d.index):
            vector = item.embedding
            if len(vector) != settings.embedding_dimensions:
                raise RuntimeError(
                    f"Embedding model {settings.embedding_model} returned "
                    f"{len(vector)} dimensions, but the schema expects "
                    f"{settings.embedding_dimensions}. Refusing to store mismatched vectors."
                )
            vectors.append(vector)
    return vectors


class EmbeddingService:
    """Explicit two-method interface, so call sites read as document vs query."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return _embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return _embed([text])[0]


embedding_service = EmbeddingService()


# Module-level helpers kept so existing call sites and tests don't churn.
def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedding_service.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    return embedding_service.embed_query(text)

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://copilot:copilot@localhost:5432/compliance_copilot"

    # --- Generation (LLM) -------------------------------------------------------
    # Provider-neutral naming: the pipeline consumes `Generation`, so the provider is
    # an implementation detail of app/llm.py. gpt-4o-mini is used because it supports
    # function calling (which is how structured output is enforced) at the cheapest
    # tier, and the task is grounded extraction over ~5 short clauses.
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("LLM_API_KEY", "OPENAI_API_KEY"),
    )

    # --- Embeddings -------------------------------------------------------------
    # One model embeds both documents and queries; see app/ingestion/embedder.py for
    # why that matters. `embedding_dimensions` is pinned to the width of the
    # `vector(N)` column -- changing it requires re-embedding the corpus and altering
    # the column, so it is deliberately explicit rather than implied by the model.
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 384
    embedding_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("EMBEDDING_API_KEY", "OPENAI_API_KEY"),
    )

    docs_dir: str = "/app/docs"

    # Candidates pulled from pgvector before filtering/reranking, vs. evidence actually
    # sent to the model. These are different numbers: filtering a final top-5 can only
    # shrink it, whereas oversampling lets a filtered-out chunk be replaced by the next
    # best applicable one.
    candidate_k: int = 20
    top_k: int = 5
    min_similarity: float = 0.25

    # --- Reranking weights ------------------------------------------------------
    # Applied on top of cosine similarity (roughly 0-1), so these are calibrated
    # against that scale. Entity applicability dominates by design; authority is a
    # gentle tiebreaker, never a universal document-type precedence order.
    rerank_entity_match_bonus: float = 0.35
    rerank_group_fallback_bonus: float = 0.05
    rerank_superseded_penalty: float = 0.60
    rerank_historical_bonus: float = 0.30
    rerank_policy_bonus: float = 0.05
    rerank_guidance_penalty: float = 0.10
    rerank_clause_bonus: float = 0.03
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # Confidence is capped by how strong the retrieved evidence actually is, not just
    # by what the model claims about itself. See app/confidence.py.
    confidence_high_similarity: float = 0.55
    confidence_medium_similarity: float = 0.40

    # gpt-4o-mini list pricing, USD per million tokens, verified against OpenAI's
    # published pricing (developers.openai.com/api/docs/pricing). Used only to attach
    # an indicative cost to each logged query -- not a billing source of truth, and it
    # will drift if OpenAI changes rates.
    price_per_mtok_input: float = 0.15
    price_per_mtok_output: float = 0.60

    # How closely a returned citation's excerpt must match the text the model was
    # actually given before we'll show it to the user. See app/verification.py.
    citation_min_match: float = 0.7

    # Upper bound on `GET /history?limit=` so a single request can't ask for the
    # entire query log.
    history_max_limit: int = 100

    @property
    def llm_key(self) -> str:
        """Key used for generation.

        Falls back to the embedding key when `LLM_API_KEY`/`OPENAI_API_KEY` is unset,
        because in this deployment both generation and embeddings run on the same
        OpenAI account. The fallback avoids storing the same secret twice under two
        names; set `LLM_API_KEY` explicitly to split the providers or rotate them
        independently.
        """
        return self.llm_api_key or self.embedding_api_key


settings = Settings()

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://copilot:copilot@localhost:5432/compliance_copilot"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5-20251001"

    embedding_model: str = "all-MiniLM-L6-v2"
    docs_dir: str = "/app/docs"

    top_k: int = 5
    min_similarity: float = 0.25
    chunk_max_tokens: int = 500
    chunk_overlap_tokens: int = 50

    # How closely a returned citation's excerpt must match the text the model was
    # actually given before we'll show it to the user. See app/verification.py.
    citation_min_match: float = 0.7

    # Upper bound on `GET /history?limit=` so a single request can't ask for the
    # entire query log.
    history_max_limit: int = 100


settings = Settings()

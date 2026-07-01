"""Central configuration, loaded from environment / `.env`.

Every component reads from a single `settings` object so the local-vs-API toggle
and the corpus definition live in exactly one place.

Note: `TICKERS`/`YEARS` are stored as plain strings and exposed as lists via properties.
pydantic-settings tries to JSON-decode `list`-typed fields from env, which would reject a
simple comma-separated value like `AAPL,MSFT` — the string-plus-property pattern avoids that.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # SEC EDGAR requires a contact identity for bulk access.
    sec_identity: str = Field(default="Anonymous example@example.com", alias="SEC_IDENTITY")

    # LLM provider: ollama | openai | azure
    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")
    ollama_model: str = Field(default="qwen2.5:3b-instruct", alias="OLLAMA_MODEL")
    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", alias="OPENAI_MODEL")
    # Azure OpenAI (the deployment name IS the model id in the API call)
    azure_openai_api_key: str | None = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: str | None = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(default="2024-05-01-preview", alias="AZURE_OPENAI_API_VERSION")
    azure_deployment: str = Field(default="gpt-oss-120b", alias="AZURE_OPENAI_DEPLOYMENT")

    # Embeddings (FastEmbed)
    embed_model: str = Field(default="BAAI/bge-base-en-v1.5", alias="EMBED_MODEL")
    sparse_model: str = Field(default="Qdrant/bm25", alias="SPARSE_MODEL")
    rerank_model: str = Field(default="BAAI/bge-reranker-base", alias="RERANK_MODEL")

    # Stores
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection: str = Field(default="tenk_chunks", alias="QDRANT_COLLECTION")
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="tenkpassword", alias="NEO4J_PASSWORD")

    # Corpus (comma-separated strings; see module docstring)
    tickers_csv: str = Field(default="AAPL,MSFT,NVDA,GOOGL,AMZN,META", alias="TICKERS")
    years_csv: str = Field(default="2022,2023,2024", alias="YEARS")
    data_dir: Path = Field(default=Path("./data"), alias="DATA_DIR")

    # Retrieval knobs
    top_k: int = Field(default=20, alias="TOP_K")
    final_k: int = Field(default=6, alias="FINAL_K")
    relevance_threshold: float = Field(default=0.45, alias="RELEVANCE_THRESHOLD")

    # ---- derived ----
    @property
    def tickers(self) -> list[str]:
        return [t.strip().upper() for t in self.tickers_csv.split(",") if t.strip()]

    @property
    def years(self) -> list[int]:
        return [int(y.strip()) for y in str(self.years_csv).split(",") if y.strip()]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

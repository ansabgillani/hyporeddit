"""Central configuration — all environment-variable access is here.

Never call os.environ directly outside this module.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    openai_api_key: str = Field(default="lm-studio", description="OpenAI API key (LM Studio ignores this)")
    llm_base_url: str = Field(default="http://localhost:1234/v1", description="Base URL for OpenAI-compatible API (LM Studio)")
    llm_provider: str = Field(default="anthropic", description="'anthropic' or 'openai'")
    # Anthropic model names
    llm_classification_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Claude model for stance classification",
    )
    llm_synthesis_model: str = Field(
        default="claude-sonnet-4-6",
        description="Claude model for synthesis",
    )
    # OpenAI-compatible model names (used when LLM_PROVIDER=openai)
    openai_classification_model: str = Field(
        default="deepseek-r1-distill-qwen-7b",
        description="OpenAI-compatible model for stance classification",
    )
    openai_synthesis_model: str = Field(
        default="deepseek-r1-distill-qwen-7b",
        description="OpenAI-compatible model for synthesis",
    )

    # Reddit source
    reddit_subreddit: str = Field(
        default="hausbau",
        description="Subreddit to ingest from (without the r/ prefix)",
    )

    # Storage
    sqlite_path: str = Field(
        default="data/sqlite/hyporeddit.db",
        description="Path to the SQLite database file",
    )
    lance_path: str = Field(
        default="data/lance",
        description="Path to the LanceDB directory",
    )

    # Embedding
    bge_m3_device: str = Field(default="cpu", description="'cpu' or 'cuda'")
    bge_m3_batch_size: int = Field(default=32, description="BGE-M3 encoding batch size")

    # Adapter
    adapter_path: str = Field(
        default="data/model/adapter.pt",
        description="Path to the linear adapter checkpoint file",
    )
    adapter_train_threshold: int = Field(
        default=200,
        description="Number of new chunks written in a process_all() run that triggers adapter training",
    )
    adapter_train_epochs: int = Field(default=3, description="Training epochs per adapter update")
    adapter_train_batch_size: int = Field(default=32, description="Pairs per gradient step")
    adapter_train_lr: float = Field(default=1e-3, description="AdamW learning rate for adapter")
    adapter_train_pairs_per_thread: int = Field(
        default=5,
        description="Max same-thread positive pairs to sample per parent_post_id",
    )

    # Ingestion
    request_delay_seconds: float = Field(
        default=1.0,
        description="Seconds to wait between HTTP requests (politeness)",
    )
    backoff_sequence: list[int] = Field(
        default=[30, 120, 600],
        description="Retry delay sequence in seconds for 429/503 responses",
    )
    max_retries: int = Field(default=3, description="Max retries per request before raising")
    circuit_breaker_threshold: float = Field(
        default=0.50,
        description="Error-rate fraction in 20-request window that triggers circuit breaker",
    )
    circuit_breaker_pause_seconds: int = Field(
        default=3600,
        description="Seconds to pause when circuit breaker fires",
    )

    # Bot/filter lists
    known_bots: list[str] = Field(
        default=["AutoModerator", "Reddit-Bot", "BotDefense"],
        description="Reddit usernames treated as bots and hard-filtered",
    )
    agreement_tokens: list[str] = Field(
        default=[
            "danke", "bitte", "+1", "👍", "^", "same", "this.", "this",
            "agreed", "stimmt", "genau", "jup", "ja.", "ok.", "okay.", "lol",
            "haha", "😂", "xd",
        ],
        description="Full-body strings that indicate pure agreement with no informational content",
    )

    # Chunking
    chunk_max_words: int = Field(default=400, description="Word count above which sub-chunking triggers")
    chunk_window_words: int = Field(default=350, description="Words per sub-chunk window")
    chunk_overlap_words: int = Field(default=50, description="Overlap words between consecutive sub-chunks")

    # Evaluation
    retrieval_top_k: int = Field(default=100, description="Number of candidate chunks to retrieve per hypothesis")
    classification_batch_size: int = Field(
        default=15, description="Chunks per stance-classification LLM call"
    )
    translation_batch_size: int = Field(
        default=5, description="Texts per translation LLM call (keep small to avoid JSON parse failures)"
    )

    # Confidence weighting constants
    recency_half_life_days: float = Field(
        default=180.0, description="Half-life for recency decay (days)"
    )
    depth_penalty_factor: float = Field(
        default=0.2, description="Per-depth-level weight penalty for nested comments"
    )
    confidence_saturation_n: int = Field(
        default=50, description="Sample size at which confidence factor saturates at 1.0"
    )

    # Logging
    log_level: str = Field(default="INFO", description="Loguru log level")


settings = Settings()


if __name__ == "__main__":
    # Run: python -m hyporeddit.config
    # Env: none required (uses .env if present)
    from loguru import logger

    logger.info("Loaded settings: llm_provider={}, sqlite_path={}, lance_path={}",
                settings.llm_provider, settings.sqlite_path, settings.lance_path)
    print(settings.model_dump())

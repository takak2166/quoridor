from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QUORIDOR_", env_file=".env", extra="ignore")

    model_hard: str = "models/quoridor_ppo_v1.zip"
    model_expert: str = "models/quoridor_ppo_best.zip"
    minimax_depth_normal: int = 3
    minimax_time_budget_normal_ms: int = 400
    minimax_max_nodes_normal: int = 1200
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    max_concurrent_games: int = 1000
    session_ttl_minutes: int = 30
    ai_concurrent_limit: int = 4
    env: str = "development"
    rate_limit_games_per_min: int = 5
    rate_limit_moves_per_min: int = 30
    trust_forwarded_for: bool = False
    require_hard_model_ready: bool = False
    require_expert_model_ready: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()

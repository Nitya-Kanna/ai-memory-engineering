from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    database_url: str = "sqlite:///./data/agent.db"
    # Episodic memory (Chroma) — separate from SQLite transcripts
    chroma_path: str = "./chroma_data"
    episodic_collection: str = "episodes"


settings = Settings()

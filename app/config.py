from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configurações lidas de variáveis de ambiente (ou do arquivo .env)."""

    database_url: str = "sqlite:///./books.db"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()

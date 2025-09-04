from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # App settings
    app_name: str = "CuraRefine"
    debug: bool = True

    # File upload settings
    max_file_size: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: list = [".csv", ".xlsx", ".xls", ".pdf", ".docx", ".txt"]
    upload_dir: str = "uploads"
    temp_dir: str = "temp"

    # AI settings
    openai_api_key: Optional[str] = None

    # Database settings (for future use)
    database_url: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()

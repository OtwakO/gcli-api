from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OAuth Configuration
    CLIENT_ID: str = (
        "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
    )
    CLIENT_SECRET: str = "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"

    # Server Configuration
    PORT: int = 7860
    DOMAIN_NAME: str = f"http://localhost:{PORT}"

    # Authentication
    GEMINI_AUTH_PASSWORD: str = "123456"

    # File Paths
    # Base path for storing persistent data, like credential files.
    PERSISTENT_STORAGE_PATH: Path = Path(__file__).parent
    # A JSON string containing a list of credential objects.
    CREDENTIALS_JSON_LIST: str = ""

    # Debugging
    DEBUG: bool = False
    DEBUG_REDACT_LOGS: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

import os

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # OAuth Configuration
    CLIENT_ID: str = (
        "681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com"
    )
    CLIENT_SECRET: str = "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"

    # Google Cloud Project Configuration
    GOOGLE_CLOUD_PROJECT: str = ""

    # Server Configuration
    PORT: int = 7860
    DOMAIN_NAME: str = f"http://localhost:{PORT}"

    # Authentication
    GEMINI_AUTH_PASSWORD: str = "123456"

    # File Paths
    PERSISTENT_STORAGE_PATH: str = os.path.dirname(os.path.abspath(__file__))
    CREDENTIAL_FILE: str = os.path.join(PERSISTENT_STORAGE_PATH, "oauth_creds.json")
    OAUTH_CREDS_JSON: str = ""

    # Debugging
    DEBUG: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

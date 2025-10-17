from pathlib import Path
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- OAuth Configuration ---
    # These are for a public Google client, but can be overridden.
    CLIENT_ID: str = Field(
        default="681255809395-oo8ft2oprdrnp9e3aqf6av3hmdib135j.apps.googleusercontent.com",
        description="The Google Cloud OAuth 2.0 Client ID.",
    )
    CLIENT_SECRET: str = Field(
        default="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",
        description="The Google Cloud OAuth 2.0 Client Secret.",
    )

    # --- Server Configuration ---
    PORT: int = Field(
        default=7860, description="The port on which the server will run."
    )
    DOMAIN_NAME: str = Field(
        default="http://localhost:7860",
        description="The public domain name of the server, used for OAuth callbacks.",
    )
    UPSTREAM_TIMEOUT: int = Field(
        default=300,
        description="Timeout in seconds for requests to the upstream Google API.",
    )
    CORS_ALLOWED_ORIGINS: List[str] = Field(
        default=["*"],
        description='A list of origins that are allowed to make cross-origin requests. Use ["*"] for public access.',
    )

    # --- API Endpoints ---
    CODE_ASSIST_ENDPOINT: str = Field(
        default="https://cloudcode-pa.googleapis.com",
        description="The endpoint for the Google Cloud Code Assist API.",
    )
    GEMINI_PUBLIC_ENDPOINT: str = Field(
        default="https://generativelanguage.googleapis.com",
        description="The endpoint for the public Google AI Gemini API.",
    )

    # --- Authentication ---
    GEMINI_AUTH_PASSWORD: str = Field(
        default="123456", description="The password required to access the proxy."
    )
    EMBEDDING_GEMINI_API_KEY: str = Field(
        default="",
        description="An optional API key for the public Gemini API, used for embeddings.",
    )

    # --- File Paths ---
    PERSISTENT_STORAGE_PATH: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent / "credentials",
        description="Base path for storing persistent data, like credential files.",
    )
    CREDENTIALS_JSON_LIST: str = Field(
        default="",
        description="A JSON string containing a list of credential objects, as an alternative to credential files.",
    )

    # --- Feature Flags & Configuration ---
    UNSUPPORTED_TOOL_SCHEMA_KEYS: List[str] = Field(
        default=["$schema", "exclusiveMinimum"],
        description="A list of JSON schema keys to be removed from tool definitions as they are not supported by the Gemini API.",
    )

    # --- Debugging ---
    DEBUG: bool = Field(
        default=False, description="Enable debug logging and other debug features."
    )
    DEBUG_REDACT_LOGS: bool = Field(
        default=True, description="Redact sensitive data from debug logs."
    )

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

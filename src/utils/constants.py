"""
Constants used across the application.
"""

import json
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)

# Client Configuration
# CLI_VERSION = "0.9.0"  # Match current gemini-cli version
CLI_VERSION = "0.22.0"  # Match current gemini-cli version

SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
]

# Default Safety Settings for Google API
DEFAULT_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "OFF"},
]


def _load_models():
    try:
        models_path = Path(__file__).parent.parent / "models.json"
        with models_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error loading models.json: {e}")
        return []


# Supported Models (for /v1beta/models endpoint)
SUPPORTED_MODELS = _load_models()

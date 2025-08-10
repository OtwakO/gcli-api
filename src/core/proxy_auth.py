import base64
import binascii

from fastapi import HTTPException, Request

from ..utils.logger import get_logger
from .settings import settings

logger = get_logger(__name__)


def authenticate_user(request: Request) -> bool:
    """Authenticate the user based on API key/password, returning True if successful."""
    # --- API Key Authentication ---
    # Gather all possible locations for an API key.
    potential_keys = [
        request.query_params.get("key"),
        request.headers.get("x-goog-api-key"),
        request.headers.get("x-api-key"),
    ]

    auth_header = request.headers.get("authorization", "")
    if auth_header.startswith("Bearer "):
        potential_keys.append(auth_header[7:])

    # Check if any provided key matches the password.
    if any(key == settings.GEMINI_AUTH_PASSWORD for key in potential_keys if key):
        return True

    # --- Basic Authentication ---
    if auth_header.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            _, password = decoded.split(":", 1)
            if password == settings.GEMINI_AUTH_PASSWORD:
                return True
        except (binascii.Error, ValueError):
            # If decoding fails, it's not valid Basic auth, so we pass
            # and fall through to the final HTTPException.
            pass

    # --- Authentication Failed ---
    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials.",
        headers={"WWW-Authenticate": "Basic"},
    )

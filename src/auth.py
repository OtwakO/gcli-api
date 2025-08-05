import base64

from fastapi import Depends, HTTPException, Request

from .settings import settings
from .logger import get_logger

logger = get_logger(__name__)


def authenticate_user(request: Request):
    """Authenticate the user based on API key/password."""
    auth_header = request.headers.get("authorization", "")
    api_key_query = request.query_params.get("key")
    goog_api_key_header = request.headers.get("x-goog-api-key")

    if (
        (api_key_query and api_key_query == settings.GEMINI_AUTH_PASSWORD)
        or (
            goog_api_key_header and goog_api_key_header == settings.GEMINI_AUTH_PASSWORD
        )
        or (
            auth_header.startswith("Bearer ")
            and auth_header[7:] == settings.GEMINI_AUTH_PASSWORD
        )
    ):
        return "api_key_user"

    if auth_header.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, password = decoded.split(":", 1)
            if password == settings.GEMINI_AUTH_PASSWORD:
                return username
        except Exception:
            pass

    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials.",
        headers={"WWW-Authenticate": "Basic"},
    )




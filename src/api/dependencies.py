from fastapi import Depends, HTTPException, Request
import json
from ..core.auth import authenticate_user
from ..core.credential_manager import ManagedCredential, get_rotating_credential
from ..utils.logger import get_logger

logger = get_logger(__name__)


async def get_request_body(request: Request) -> dict:
    """FastAPI dependency to parse and return the JSON body of a request."""
    try:
        post_data = await request.body()
        return json.loads(post_data) if post_data else {}
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid JSON in request body: {e}"
        )


async def get_validated_credential(
    request: Request,
    username: str = Depends(authenticate_user),
    managed_cred: ManagedCredential = Depends(get_rotating_credential),
) -> ManagedCredential:
    """
    A FastAPI dependency that handles user authentication and credential rotation.
    It ensures that a valid, ready-to-use credential is available.
    """
    if not managed_cred:
        logger.error("No managed credential was returned from the credential manager.")
        raise HTTPException(
            status_code=503,
            detail="No valid credentials available in the rotation pool.",
        )
    return managed_cred

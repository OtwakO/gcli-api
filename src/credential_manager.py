import asyncio
import json
import logging
import re
from typing import List, Optional

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from pydantic import BaseModel, Field
from fastapi import HTTPException

from .settings import settings
from .constants import SCOPES
from .logger import get_logger

logger = get_logger(__name__)

class ManagedCredential(BaseModel):
    """Encapsulates a credential and its associated state."""
    credential: Credentials
    project_id: Optional[str] = None
    user_email: Optional[str] = None
    is_onboarded: bool = False
    is_valid: bool = Field(True, exclude=True)  # Exclude from serialization

    class Config:
        arbitrary_types_allowed = True

class CredentialManager:
    """Loads, manages, and rotates a pool of OAuth credentials."""
    def __init__(self):
        self._credentials: List[ManagedCredential] = []
        self._lock = asyncio.Lock()
        self._next_credential_index: int = 0
        self.load_credentials()

    def load_credentials(self):
        """Loads credentials using the hybrid priority system."""
        if self._load_from_env():
            logger.info(f"Loaded {len(self._credentials)} credentials from CREDENTIALS_JSON_LIST env var.")
        else:
            self._load_from_files()
            if self._credentials:
                logger.info(f"Loaded {len(self._credentials)} credentials from oauth_creds_*.json files.")

    def _load_from_env(self) -> bool:
        creds_json_list_str = settings.CREDENTIALS_JSON_LIST
        if not creds_json_list_str:
            return False
        
        try:
            creds_list = json.loads(creds_json_list_str)
            if not isinstance(creds_list, list):
                raise ValueError("CREDENTIALS_JSON_LIST must be a JSON array.")
            
            for cred_info in creds_list:
                self._add_credential_from_info(cred_info)
            return True
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Failed to parse CREDENTIALS_JSON_LIST: {e}")
            return False

    def _load_from_files(self):
        """Loads credentials from .json files in the persistent storage path."""
        storage_path = settings.PERSISTENT_STORAGE_PATH
        for file_path in storage_path.glob("oauth_creds_*.json"):
            try:
                with file_path.open("r", encoding="utf-8") as f:
                    cred_info = json.load(f)
                    self._add_credential_from_info(cred_info, str(file_path))
            except Exception as e:
                logger.warning(f"Could not load credential file {file_path}: {e}")

    def _add_credential_from_info(self, cred_info: dict, source: str = "env"):
        if "refresh_token" not in cred_info:
            logger.warning(f"Skipping a credential from {source} due to missing 'refresh_token'.")
            return

        minimal_creds_data = {
            "client_id": cred_info.get("client_id", settings.CLIENT_ID),
            "client_secret": cred_info.get("client_secret", settings.CLIENT_SECRET),
            "refresh_token": cred_info["refresh_token"],
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        google_creds = Credentials.from_authorized_user_info(minimal_creds_data, SCOPES)
        project_id = cred_info.get("project_id")
        user_email = cred_info.get("user_email")
        
        self._credentials.append(
            ManagedCredential(
                credential=google_creds,
                project_id=project_id,
                user_email=user_email,
            )
        )

    async def get_next_credential(self) -> Optional[ManagedCredential]:
        """Rotates and returns the next valid and refreshed credential."""
        if not self._credentials:
            return None

        async with self._lock:
            # Loop through all available credentials once to find a valid one
            for _ in range(len(self._credentials)):
                index = self._next_credential_index
                managed_cred = self._credentials[index]
                self._next_credential_index = (self._next_credential_index + 1) % len(self._credentials)

                # Skip credentials that have been marked as invalid
                if not managed_cred.is_valid:
                    continue

                if settings.DEBUG:
                    email = managed_cred.user_email or "unknown"
                    token_snippet = managed_cred.credential.refresh_token[-4:]
                    logger.debug(f"Rotating to credential index {index} (User: {email}, Token ends in: ...{token_snippet})")

                # If the credential is not expired, it's good to use immediately
                if not managed_cred.credential.expired:
                    return managed_cred

                # If it's expired, try to refresh it
                if managed_cred.credential.refresh_token:
                    try:
                        logger.info(f"Credential for {managed_cred.user_email} expired. Refreshing...")
                        await asyncio.to_thread(managed_cred.credential.refresh, GoogleAuthRequest())
                        logger.info(f"Credential for {managed_cred.user_email} refreshed successfully.")
                        return managed_cred  # Return the now-refreshed credential
                    except RefreshError as e:
                        logger.error(
                            f"Failed to refresh credential for {managed_cred.user_email}. "
                            f"This credential will be marked as invalid for this session. Error: {e}"
                        )
                        managed_cred.is_valid = False  # Mark as invalid
                        continue  # Try the next credential
                    except Exception as e:
                        logger.error(f"An unexpected error occurred while refreshing credential for {managed_cred.user_email}: {e}")
                        # Don't mark as invalid for transient errors, just try the next one
                        continue

        # If we've looped through all credentials and found no valid ones
        logger.error("No valid credentials available in the pool after checking all of them.")
        return None



credential_manager = CredentialManager()

async def get_rotating_credential() -> ManagedCredential:
    """FastAPI dependency to get the next credential from the pool."""
    managed_cred = await credential_manager.get_next_credential()
    if not managed_cred:
        raise HTTPException(
            status_code=503,
            detail="No valid credentials available in the rotation pool. Please run the generator or configure environment variables."
        )
    return managed_cred

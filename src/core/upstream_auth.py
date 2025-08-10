from abc import ABC, abstractmethod
from typing import Dict

import httpx

from .credential_manager import ManagedCredential


class AuthStrategy(ABC):
    """Abstract base class for an authentication strategy."""

    @abstractmethod
    def get_headers(self) -> Dict[str, str]:
        """Returns the authentication headers for the request."""
        pass

    def prepare_client(self, client: httpx.AsyncClient) -> httpx.AsyncClient:
        """
        Allows the strategy to modify the httpx.AsyncClient if needed
        (e.g., for OAuth 2.0 flows). By default, does nothing.
        """
        return client


class OAuthStrategy(AuthStrategy):
    """Authentication using an OAuth 2.0 bearer token."""

    def __init__(self, managed_cred: ManagedCredential):
        if not managed_cred.credential or not managed_cred.credential.token:
            raise ValueError("OAuth credential is missing a valid token.")
        self.token = managed_cred.credential.token

    def get_headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


class ApiKeyStrategy(AuthStrategy):
    """Authentication using an API key in the query parameters."""

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key cannot be empty.")
        self.api_key = api_key

    def get_headers(self) -> Dict[str, str]:
        # API key is sent as a query param, so no extra headers are needed.
        return {}

    def prepare_client(self, client: httpx.AsyncClient) -> httpx.AsyncClient:
        """Adds the API key as a default query parameter to the client."""
        client.params = client.params.set("key", self.api_key)
        return client

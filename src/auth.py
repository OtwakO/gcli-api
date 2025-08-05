import asyncio
import base64
import json
import logging
import os

import httpx
from fastapi import Depends, HTTPException, Request

from .constants import CODE_ASSIST_ENDPOINT
from .credential_manager import ManagedCredential
from .settings import settings
from .utils import get_client_metadata, get_user_agent


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


async def onboard_user(managed_cred: ManagedCredential):
    """Ensures the user associated with the credential is onboarded."""
    if managed_cred.is_onboarded:
        return

    creds = managed_cred.credential
    project_id = managed_cred.project_id

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }
    load_assist_payload = {
        "cloudaicompanionProject": project_id,
        "metadata": get_client_metadata(project_id),
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
                data=json.dumps(load_assist_payload, ensure_ascii=False),
                headers=headers,
            )
            resp.raise_for_status()
            load_data = resp.json()

            if load_data.get("currentTier"):
                managed_cred.is_onboarded = True
                return

            tier = next(
                (t for t in load_data.get("allowedTiers", []) if t.get("isDefault")),
                {"id": "legacy-tier"},
            )

            onboard_req_payload = {
                "tierId": tier.get("id"),
                "cloudaicompanionProject": project_id,
                "metadata": get_client_metadata(project_id),
            }

            while True:
                onboard_resp = await client.post(
                    f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
                    data=json.dumps(onboard_req_payload, ensure_ascii=False),
                    headers=headers,
                )
                onboard_resp.raise_for_status()
                lro_data = onboard_resp.json()
                if lro_data.get("done"):
                    managed_cred.is_onboarded = True
                    break
                await asyncio.sleep(5)
    except httpx.HTTPStatusError as e:
        logging.error(
            f"Error during onboarding for project {project_id}: {e.response.text}"
        )
        raise


async def get_user_project_id(managed_cred: ManagedCredential) -> str:
    """Gets project ID for the given credential, using cache if available."""
    if managed_cred.project_id:
        return managed_cred.project_id

    creds = managed_cred.credential
    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
                data=json.dumps(
                    {"metadata": get_client_metadata()}, ensure_ascii=False
                ),
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            api_project_id = data.get("cloudaicompanionProject")
            if not api_project_id:
                raise ValueError(
                    "Could not find 'cloudaicompanionProject' from API response."
                )

            managed_cred.project_id = api_project_id
            return api_project_id
    except Exception as e:
        logging.error(f"Could not discover project ID from API: {e}")
        raise

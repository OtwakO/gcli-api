import asyncio
import json
import random
from typing import Dict, Any

import httpx
from fastapi import HTTPException
from pydantic import ValidationError

from .credential_manager import ManagedCredential
from .constants import DEFAULT_SAFETY_SETTINGS
from .models import GeminiRequest
from .settings import settings
from .utils import get_user_agent, create_redacted_payload, get_client_metadata
from .logger import get_logger, format_log

logger = get_logger(__name__)


async def _get_user_project_id(managed_cred: ManagedCredential) -> str:
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
                f"{settings.CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
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
            logger.info(f"Discovered project ID for {managed_cred.user_email}: {api_project_id}")
            return api_project_id
    except (httpx.HTTPStatusError, ValueError, json.JSONDecodeError) as e:
        logger.error(f"Could not discover project ID from API for {managed_cred.user_email}: {e}")
        raise HTTPException(status_code=500, detail="Failed to discover user project ID from Google API.")


async def _onboard_user(managed_cred: ManagedCredential):
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
                f"{settings.CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
                data=json.dumps(load_assist_payload, ensure_ascii=False),
                headers=headers,
            )
            resp.raise_for_status()
            load_data = resp.json()

            if load_data.get("currentTier"):
                managed_cred.is_onboarded = True
                logger.info(f"User {managed_cred.user_email} is already onboarded for project {project_id}.")
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

            logger.info(f"Onboarding user {managed_cred.user_email} for project {project_id}...")
            # This is a long-running operation (LRO), so we poll with exponential backoff.
            max_attempts = 5
            base_delay = 1.0  # seconds
            for attempt in range(max_attempts):
                onboard_resp = await client.post(
                    f"{settings.CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
                    data=json.dumps(onboard_req_payload, ensure_ascii=False),
                    headers=headers,
                )
                onboard_resp.raise_for_status()
                lro_data = onboard_resp.json()
                if lro_data.get("done"):
                    managed_cred.is_onboarded = True
                    logger.info(f"Successfully onboarded user {managed_cred.user_email}.")
                    return
                
                # Exponential backoff with jitter
                delay = (base_delay * 2 ** attempt) + (random.uniform(0, 1))
                logger.info(f"Onboarding not complete, retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
            
            logger.warning(f"Onboarding LRO for {managed_cred.user_email} did not complete after {max_attempts} attempts.")

    except httpx.HTTPStatusError as e:
        logger.error(
            f"Error during onboarding for project {project_id}: {e.response.text}"
        )
        raise HTTPException(status_code=e.response.status_code, detail=f"Onboarding failed: {e.response.text}")


async def send_gemini_request(
    managed_cred: ManagedCredential, payload: dict, is_streaming: bool = False
) -> httpx.Response:
    """
    Prepares and sends a request to the Google Gemini API, handling auth, project discovery, and onboarding.
    """
    creds = managed_cred.credential
    if not creds or not creds.token:
        raise HTTPException(status_code=401, detail="Credential is missing a valid token.")

    # Ensure project ID is known and user is onboarded
    proj_id = await _get_user_project_id(managed_cred)
    await _onboard_user(managed_cred)

    action = "streamGenerateContent" if is_streaming else "generateContent"
    target_url = f"{settings.CODE_ASSIST_ENDPOINT}/v1internal:{action}"
    if is_streaming:
        target_url += "?alt=sse"

    request_headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    final_payload = {
        "model": payload.get("model"),
        "project": proj_id,
        "request": payload.get("request", {}),
    }
    final_post_data = json.dumps(final_payload, ensure_ascii=False)

    if settings.DEBUG:
        log_payload = (
            create_redacted_payload(final_payload)
            if settings.DEBUG_REDACT_LOGS
            else final_payload
        )
        logger.debug(
            format_log(
                "Upstream Request to Google",
                {"url": target_url, "headers": request_headers, "payload": log_payload},
                is_json=True,
            )
        )

    try:
        async with httpx.AsyncClient(timeout=settings.UPSTREAM_TIMEOUT) as client:
            response = await client.post(
                target_url, data=final_post_data, headers=request_headers
            )
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        error_body = e.response.text
        try:
            # Try to parse and format the error as JSON for readability
            error_body = e.response.json()
        except json.JSONDecodeError:
            pass  # Keep as raw text if not JSON
        
        logger.error(format_log(
            f"Upstream API Error ({e.response.status_code})",
            error_body,
            is_json=isinstance(error_body, dict)
        ))
        raise HTTPException(status_code=e.response.status_code, detail=e.response.text)

    if settings.DEBUG:
        log_data = {"status_code": response.status_code, "headers": dict(response.headers)}
        if not is_streaming:
            try:
                log_data["body"] = response.json()
            except json.JSONDecodeError:
                log_data["body"] = response.text
        logger.debug(format_log("Upstream Response from Google", log_data, is_json=True))

    return response



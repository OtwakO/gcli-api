import asyncio
import json
import random
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict

import httpx
from fastapi import HTTPException

from ..models.gemini import GeminiResponse
from ..utils.logger import format_log, get_logger
from ..utils.utils import (
    create_redacted_payload,
    get_client_metadata,
    get_user_agent,
    summarize_embedding_logs,
)
from .credential_manager import ManagedCredential
from .settings import settings
from .streaming import Streamer, _parse_google_sse

logger = get_logger(__name__)


@dataclass
class PreparedRequest:
    """Holds the components for making an upstream API request."""

    headers: Dict[str, str]
    data: str


def _prepare_request_components(
    managed_cred: ManagedCredential, payload: Dict[str, Any], target_url: str
) -> PreparedRequest:
    """
    Prepares headers and serialized data for a Google API request.
    Also handles debug logging.
    """
    creds = managed_cred.credential
    if not creds or not creds.token:
        raise HTTPException(
            status_code=401, detail="Credential is missing a valid token."
        )

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    final_post_data = json.dumps(payload, ensure_ascii=False)

    if settings.DEBUG:
        log_payload = (
            create_redacted_payload(payload) if settings.DEBUG_REDACT_LOGS else payload
        )
        logger.debug(
            format_log(
                "Upstream Request to Google",
                {"url": target_url, "headers": headers, "payload": log_payload},
                is_json=True,
            )
        )

    return PreparedRequest(headers=headers, data=final_post_data)


class GoogleStreamer(Streamer):
    """Streams responses from the Google Gemini API."""

    def __init__(
        self,
        managed_cred: ManagedCredential,
        target_url: str,
        payload: Dict[str, Any],
    ):
        self.managed_cred = managed_cred
        self.target_url = target_url
        self.payload = payload

    async def stream(self) -> AsyncGenerator[GeminiResponse, None]:
        prepared_request = _prepare_request_components(
            self.managed_cred, self.payload, self.target_url
        )

        async with httpx.AsyncClient(timeout=settings.UPSTREAM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                self.target_url,
                headers=prepared_request.headers,
                data=prepared_request.data,
            ) as response:
                response.raise_for_status()
                if settings.DEBUG:
                    logger.debug(
                        f"Upstream Response from Google (Streaming): {response.status_code}"
                    )
                async for chunk in _parse_google_sse(response):
                    yield chunk


async def _fetch_project_id(managed_cred: ManagedCredential) -> str:
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
            logger.info(
                f"Discovered project ID for {managed_cred.user_email}: {api_project_id}"
            )
            return api_project_id
    except (httpx.HTTPStatusError, ValueError, json.JSONDecodeError) as e:
        logger.error(
            f"Could not discover project ID from API for {managed_cred.user_email}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to discover user project ID from Google API.",
        )


async def _perform_onboarding(managed_cred: ManagedCredential, project_id: str):
    """Ensures the user associated with the credential is onboarded."""
    if managed_cred.is_onboarded:
        return

    creds = managed_cred.credential

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
                logger.info(
                    f"User {managed_cred.user_email} is already onboarded for project {project_id}."
                )
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

            logger.info(
                f"Onboarding user {managed_cred.user_email} for project {project_id}..."
            )
            max_attempts = 5
            base_delay = 1.0
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
                    logger.info(
                        f"Successfully onboarded user {managed_cred.user_email}."
                    )
                    return

                delay = (base_delay * 2**attempt) + (random.uniform(0, 1))
                logger.info(
                    f"Onboarding not complete, retrying in {delay:.2f} seconds..."
                )
                await asyncio.sleep(delay)

            logger.warning(
                f"Onboarding LRO for {managed_cred.user_email} did not complete after {max_attempts} attempts."
            )

    except httpx.HTTPStatusError as e:
        logger.error(
            f"Error during onboarding for project {project_id}: {e.response.text}"
        )
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Onboarding failed: {e.response.text}",
        )


async def prepare_credential(managed_cred: ManagedCredential) -> str:
    """
    Ensures the credential has a project ID and the user is onboarded.
    Returns the project ID.
    """
    project_id = managed_cred.project_id
    if not project_id:
        project_id = await _fetch_project_id(managed_cred)

    if not managed_cred.is_onboarded:
        await _perform_onboarding(managed_cred, project_id)

    return project_id


async def send_gemini_request(
    managed_cred: ManagedCredential, target_url: str, payload: dict
) -> httpx.Response:
    """Sends a non-streaming, OAuth-authenticated request to the Google Gemini API."""
    prepared_request = _prepare_request_components(managed_cred, payload, target_url)

    async with httpx.AsyncClient(timeout=settings.UPSTREAM_TIMEOUT) as client:
        try:
            response = await client.post(
                target_url, data=prepared_request.data, headers=prepared_request.headers
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            try:
                error_body = e.response.json()
            except json.JSONDecodeError:
                pass
            logger.error(
                format_log(
                    f"Upstream API Error ({e.response.status_code})",
                    error_body,
                    is_json=isinstance(error_body, dict),
                )
            )
            raise HTTPException(
                status_code=e.response.status_code, detail=e.response.text
            )

    if settings.DEBUG:
        log_data = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }
        try:
            body_json = response.json()
            # Summarize embedding values for cleaner logs
            log_data["body"] = summarize_embedding_logs(body_json)
        except json.JSONDecodeError:
            log_data["body"] = response.text
        logger.debug(
            format_log("Upstream Response from Google", log_data, is_json=True)
        )

    return response


async def send_public_api_request(
    target_url: str, api_key: str, payload: dict
) -> httpx.Response:
    """Sends a non-streaming, API Key-authenticated request to a public Google API."""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }
    # The API key is sent as a query parameter
    url_with_key = f"{target_url}?key={api_key}"
    final_post_data = json.dumps(payload, ensure_ascii=False)

    if settings.DEBUG:
        log_payload = (
            create_redacted_payload(payload) if settings.DEBUG_REDACT_LOGS else payload
        )
        logger.debug(
            format_log(
                "Upstream Request to Google (Public)",
                {"url": url_with_key, "headers": headers, "payload": log_payload},
                is_json=True,
            )
        )

    async with httpx.AsyncClient(timeout=settings.UPSTREAM_TIMEOUT) as client:
        try:
            response = await client.post(
                url_with_key, data=final_post_data, headers=headers
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            try:
                error_body = e.response.json()
            except json.JSONDecodeError:
                pass
            logger.error(
                format_log(
                    f"Upstream API Error ({e.response.status_code})",
                    error_body,
                    is_json=isinstance(error_body, dict),
                )
            )
            raise HTTPException(
                status_code=e.response.status_code, detail=e.response.text
            )

    if settings.DEBUG:
        log_data = {
            "status_code": response.status_code,
            "headers": dict(response.headers),
        }
        try:
            body_json = response.json()
            # Summarize embedding values for cleaner logs
            log_data["body"] = summarize_embedding_logs(body_json)
        except json.JSONDecodeError:
            log_data["body"] = response.text
        logger.debug(
            format_log("Upstream Response from Google", log_data, is_json=True)
        )

    return response

import json
from typing import Any, Dict

import httpx
from fastapi import HTTPException

from ..utils.logger import format_log, get_logger
from ..utils.utils import (
    create_redacted_payload,
    get_user_agent,
    summarize_embedding_logs,
)
from .upstream_auth import AuthStrategy
from .settings import settings

logger = get_logger(__name__)


async def send_request(
    target_url: str, payload: Dict[str, Any], auth_strategy: AuthStrategy
) -> httpx.Response:
    """
    Sends a non-streaming, authenticated request to a Google API using a specified
    authentication strategy.
    """
    headers = {
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
        **auth_strategy.get_headers(),  # Add auth-specific headers
    }

    final_post_data = json.dumps(payload, ensure_ascii=False)

    if settings.DEBUG:
        log_payload = (
            create_redacted_payload(payload) if settings.DEBUG_REDACT_LOGS else payload
        )
        log_title = f"Upstream Request to Google ({type(auth_strategy).__name__})"
        logger.debug(
            format_log(
                log_title,
                {"url": target_url, "headers": headers, "payload": log_payload},
                is_json=True,
            )
        )

    async with httpx.AsyncClient(timeout=settings.UPSTREAM_TIMEOUT) as client:
        # Allow the strategy to modify the client (e.g., add query params)
        client = auth_strategy.prepare_client(client)

        try:
            response = await client.post(
                target_url, data=final_post_data, headers=headers
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            try:
                error_body = e.response.json()
            except json.JSONDecodeError:
                pass  # Keep body as text if not valid JSON

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
            log_data["body"] = summarize_embedding_logs(body_json)
        except json.JSONDecodeError:
            log_data["body"] = response.text
        logger.debug(
            format_log("Upstream Response from Google", log_data, is_json=True)
        )

    return response

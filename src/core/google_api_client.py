import json
from typing import Any, Dict

import httpx

from ..core.exceptions import UpstreamHttpError

from ..utils.logger import format_log, get_logger, log_upstream_request
from ..utils.utils import (
    get_user_agent,
    summarize_embedding_logs,
)
from .settings import settings
from .upstream_auth import AuthStrategy

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

    # Use the centralized logger to log the outgoing request.
    log_upstream_request(
        url=target_url,
        headers=headers,
        payload=payload,
        auth_strategy_name=type(auth_strategy).__name__,
    )

    async with httpx.AsyncClient(timeout=settings.UPSTREAM_TIMEOUT) as client:
        # Allow the strategy to modify the client (e.g., add query params)
        client = auth_strategy.prepare_client(client)

        try:
            final_post_data = json.dumps(payload, ensure_ascii=False)
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

            log_message = format_log(
                f"Upstream API Error ({e.response.status_code})",
                error_body,
                is_json=isinstance(error_body, dict),
            )
            logger.warning(log_message)
            raise UpstreamHttpError(status_code=e.response.status_code, detail=error_body)

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

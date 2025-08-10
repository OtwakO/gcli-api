import json
from typing import Any, AsyncGenerator, Dict, Union

import httpx
from pydantic import ValidationError

from ..adapters.formatters import Formatter, OpenAIFormatter
from .upstream_auth import OAuthStrategy
from ..core.credential_manager import ManagedCredential
from ..models.gemini import GeminiResponse
from ..utils.logger import format_log, get_logger
from ..utils.utils import get_user_agent
from .settings import settings

logger = get_logger(__name__)


async def _parse_google_sse(
    response: httpx.Response,
) -> AsyncGenerator[GeminiResponse, None]:
    """
    Parses Google's SSE stream line-by-line.
    """
    async for line in response.aiter_lines():
        if not line:
            continue
        line = line.strip()
        if not line.startswith("data:"):
            continue

        data_str = line[len("data:") :].strip()
        if not data_str:
            continue

        try:
            api_response_obj = json.loads(data_str)
            gemini_response = None

            try:
                gemini_response = GeminiResponse.model_validate(api_response_obj)
            except ValidationError:
                if "response" in api_response_obj:
                    gemini_response = GeminiResponse.model_validate(
                        api_response_obj["response"]
                    )
                    if (
                        "usageMetadata" in api_response_obj
                        and not gemini_response.usageMetadata
                    ):
                        gemini_response.usageMetadata = api_response_obj.get(
                            "usageMetadata"
                        )

            if gemini_response:
                if settings.DEBUG:
                    logger.debug(
                        format_log(
                            "Processed Chunk from Upstream",
                            gemini_response.model_dump(exclude_unset=True),
                            is_json=True,
                        )
                    )
                yield gemini_response
            elif "usageMetadata" in api_response_obj:
                gemini_response = GeminiResponse(
                    candidates=[], usageMetadata=api_response_obj["usageMetadata"]
                )
                if settings.DEBUG:
                    logger.debug(
                        format_log(
                            "Processed Chunk from Upstream (Metadata Only)",
                            gemini_response.model_dump(exclude_unset=True),
                            is_json=True,
                        )
                    )
                yield gemini_response

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(
                f"Skipping a malformed SSE chunk. Error: {e}. Chunk: '{data_str}'"
            )


class StreamError(Exception):
    """Represents a handled error from the stream."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class StreamProcessor:
    """
    Handles the entire streaming process, from making the request to formatting
    the output and handling errors gracefully.
    """

    def __init__(
        self,
        managed_cred: ManagedCredential,  # Kept for now, auth_strategy is primary
        target_url: str,
        payload: Dict[str, Any],
        formatter: Formatter,
    ):
        self.target_url = target_url
        self.payload = payload
        self.formatter = formatter
        self.auth_strategy = OAuthStrategy(managed_cred)

    async def _stream_generator(
        self,
    ) -> AsyncGenerator[Union[GeminiResponse, StreamError], None]:
        """
        Connects to the upstream API and yields either GeminiResponse chunks
        or a StreamError.
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": get_user_agent(),
            **self.auth_strategy.get_headers(),
        }
        final_post_data = json.dumps(self.payload, ensure_ascii=False)

        async with httpx.AsyncClient(timeout=settings.UPSTREAM_TIMEOUT) as client:
            async with client.stream(
                "POST",
                self.target_url,
                headers=headers,
                data=final_post_data,
            ) as response:
                try:
                    response.raise_for_status()
                    async for chunk in _parse_google_sse(response):
                        yield chunk
                except httpx.HTTPStatusError as e:
                    error_body = await e.response.aread()
                    yield StreamError(
                        status_code=e.response.status_code, message=error_body.decode()
                    )

    async def process(self) -> AsyncGenerator[str, None]:
        """
        The main pipeline orchestrator. It consumes the stream generator and
        formats the output for the client, handling errors cleanly.
        """
        had_error = False
        try:
            async for chunk in self._stream_generator():
                if isinstance(chunk, StreamError):
                    had_error = True
                    logger.error(
                        f"Upstream API Error (Streaming): {chunk.status_code} - {chunk.message}"
                    )
                    error_payload = {"error": {"message": chunk.message}}
                    yield f"data: {json.dumps(error_payload)}\n\n"
                    break  # Stop processing after a handled error
                else:
                    for event in self.formatter.format_chunk(chunk):
                        yield event

            if not had_error:
                # Signal the end of the stream to the formatter
                for event in self.formatter.format_chunk(None):
                    yield event

        except Exception as e:
            # This catches unexpected errors (bugs, network issues, etc.)
            logger.error("Generic stream processing error", exc_info=True)
            error_payload = {
                "error": {
                    "message": f"An unexpected error occurred during streaming: {e}"
                }
            }
            yield f"data: {json.dumps(error_payload)}\n\n"

        if isinstance(self.formatter, OpenAIFormatter):
            yield "data: [DONE]\n\n"

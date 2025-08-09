import json
from abc import ABC, abstractmethod
from typing import AsyncGenerator

import httpx
from pydantic import ValidationError

from ..adapters.formatters import Formatter, OpenAIFormatter
from ..models.gemini import GeminiResponse
from ..utils.logger import format_log, get_logger
from .settings import settings

logger = get_logger(__name__)


class StreamParseError(Exception):
    """Custom exception for errors during stream parsing."""

    def __init__(self, message: str, original_text: str):
        super().__init__(message)
        self.original_text = original_text


async def _parse_google_sse(
    response: httpx.Response,
) -> AsyncGenerator[GeminiResponse, None]:
    """
    Parses Google's SSE stream line-by-line, mirroring the logic from the
    working reference project.
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

            # Strategy 1: Try to parse the entire object as a GeminiResponse.
            try:
                gemini_response = GeminiResponse.model_validate(api_response_obj)
            except ValidationError:
                # Strategy 2: If that fails, look for a "response" key.
                if "response" in api_response_obj:
                    gemini_response = GeminiResponse.model_validate(
                        api_response_obj["response"]
                    )
                    # Carry over usage metadata if it's outside the 'response' object
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
            # Strategy 3: Handle metadata-only chunks.
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


class Streamer(ABC):
    """Abstract base class for a provider-specific streamer."""

    @abstractmethod
    async def stream(self) -> AsyncGenerator[GeminiResponse, None]:
        """Yields GeminiResponse objects from the upstream provider."""
        yield


class StreamProcessor:
    """Orchestrates the streaming process by connecting a Streamer to a Formatter."""

    def __init__(self, streamer: Streamer, formatter: Formatter):
        self.streamer = streamer
        self.formatter = formatter

    async def process(self) -> AsyncGenerator[str, None]:
        """The main pipeline orchestrator for processing and formatting the stream."""
        try:
            async for chunk in self.streamer.stream():
                for event in self.formatter.format_chunk(chunk):
                    yield event

            # Signal the end of the stream to the formatter
            for event in self.formatter.format_chunk(None):
                yield event

        except Exception as e:
            logger.error(f"Generic stream processing error: {e}", exc_info=True)
            error_payload = {
                "error": {
                    "message": f"An unexpected error occurred during streaming: {e}"
                }
            }
            yield f"data: {json.dumps(error_payload)}"

        if isinstance(self.formatter, OpenAIFormatter):
            yield "data: [DONE]"

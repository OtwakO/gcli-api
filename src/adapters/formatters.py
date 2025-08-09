from typing import Any, Dict, Generator, Optional, Union

from pydantic import BaseModel

from ..models.gemini import (
    BatchEmbedContentResponse,
    EmbedContentResponse,
    GeminiResponse,
)
from .claude_transformers import ClaudeStreamer, gemini_response_to_claude
from .openai_transformers import (
    gemini_response_to_openai,
    gemini_response_to_openai_embedding,
    gemini_stream_chunk_to_openai,
)

# --- Formatter Classes ---


class Formatter:
    """Base class for all formatters."""

    def __init__(self, context: Dict[str, Any]):
        self.context = context

    def format_chunk(
        self,
        chunk: Optional[GeminiResponse],
    ) -> Generator[str, None, None]:
        """Formats a single chunk for a streaming response."""
        raise NotImplementedError

    def format_response(
        self,
        response: Any,
        original_request: Any,
    ) -> BaseModel:
        """Formats a complete, non-streaming GeminiResponse."""
        return response


class GeminiFormatter(Formatter):
    """Formats responses for the native Gemini API."""

    def format_chunk(
        self,
        chunk: Optional[GeminiResponse],
    ) -> Generator[str, None, None]:
        if chunk:
            yield f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"


class OpenAIFormatter(Formatter):
    """Formats responses for the OpenAI-compatible API."""

    def format_chunk(
        self,
        chunk: Optional[GeminiResponse],
    ) -> Generator[str, None, None]:
        if chunk:
            openai_chunk = gemini_stream_chunk_to_openai(
                chunk, self.context["model"], self.context["response_id"]
            )
            yield f"data: {openai_chunk.model_dump_json(exclude_unset=True)}\n\n"

    def format_response(
        self,
        response: GeminiResponse,
        original_request: Any,
    ) -> BaseModel:
        return gemini_response_to_openai(response, original_request)


class OpenAIEmbeddingFormatter(Formatter):
    """Formats responses for the OpenAI-compatible embedding API."""

    def format_response(
        self,
        response: Union[EmbedContentResponse, BatchEmbedContentResponse],
        original_request: Any,
    ) -> BaseModel:
        return gemini_response_to_openai_embedding(response, original_request)


class ClaudeFormatter(Formatter):
    """Stateful formatter for the Claude API event stream."""

    def __init__(self, context: Dict[str, Any]):
        super().__init__(context)
        self.streamer = ClaudeStreamer(
            response_id=self.context["response_id"],
            model=self.context["model"],
        )

    def format_chunk(
        self,
        chunk: Optional[GeminiResponse],
    ) -> Generator[str, None, None]:
        yield from self.streamer.format_chunk(chunk)

    def format_response(
        self,
        response: GeminiResponse,
        original_request: Any,
    ) -> BaseModel:
        return gemini_response_to_claude(response, original_request)

import json
from typing import Any, Generator, Optional, Union

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


class FormatterContext(BaseModel):
    """A structured container for context needed by formatters."""

    response_id: str
    model: str


# --- Formatter Classes ---


class Formatter:
    """Base class for all formatters."""

    def __init__(self, context: FormatterContext):
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

    def format_error_chunk(self, message: str, status_code: int = 500) -> str:
        """Formats a generic error message into an SSE event string."""
        error_payload = {"error": {"message": message, "code": status_code}}
        return f"data: {json.dumps(error_payload)}\n\n"


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

    def __init__(self, context: FormatterContext):
        super().__init__(context)
        # Initialize with fallback values from the context
        self.response_id = self.context.response_id
        self.model = self.context.model
        self.meta_data_captured = False

    def format_chunk(
        self,
        chunk: Optional[GeminiResponse],
    ) -> Generator[str, None, None]:
        # Try to capture metadata from the stream until we succeed
        if chunk and not self.meta_data_captured:
            if chunk.responseId:
                self.response_id = f"chatcmpl-{chunk.responseId}"
            if chunk.modelVersion:
                self.model = chunk.modelVersion
            # Once we see a responseId, we assume all initial metadata is captured.
            if chunk.responseId:
                self.meta_data_captured = True

        if chunk:
            # Pass the potentially updated model and response_id to the transformer
            openai_chunk = gemini_stream_chunk_to_openai(
                chunk, self.model, self.response_id
            )
            yield f"data: {openai_chunk.model_dump_json(exclude_unset=True)}\n\n"

    def format_error_chunk(self, message: str, status_code: int = 500) -> str:
        """Formats an error message into an OpenAI-compatible SSE data chunk."""
        # OpenAI errors typically have this structure
        error_payload = {
            "error": {"message": message, "type": "server_error", "code": None}
        }
        return f"data: {json.dumps(error_payload)}\n\n"

    def format_response(
        self,
        response: GeminiResponse,
        original_request: Any,
    ) -> BaseModel:
        return gemini_response_to_openai(response, original_request)


class OpenAIEmbeddingFormatter(Formatter):
    """Formats responses for the OpenAI-compatible embedding API."""

    def __init__(self, context: FormatterContext):
        # This formatter doesn't use context, but we accept it for consistency
        super().__init__(context)

    def format_response(
        self,
        response: Union[EmbedContentResponse, BatchEmbedContentResponse],
        original_request: Any,
    ) -> BaseModel:
        return gemini_response_to_openai_embedding(response, original_request)


class ClaudeFormatter(Formatter):
    """Stateful formatter for the Claude API event stream."""

    def __init__(self, context: FormatterContext):
        super().__init__(context)
        # Initialize the streamer with fallback values
        self.streamer = ClaudeStreamer(
            response_id=self.context.response_id,
            model=self.context.model,
        )

    def format_chunk(
        self,
        chunk: Optional[GeminiResponse],
    ) -> Generator[str, None, None]:
        # Before processing the chunk, check if we can capture metadata from it.
        if chunk and not self.streamer.meta_data_captured:
            if chunk.responseId:
                # Update the streamer's response_id before it's used in the first event
                self.streamer.response_id = f"msg_{chunk.responseId}"
            if chunk.modelVersion:
                self.streamer.model = chunk.modelVersion

            # If we found an ID, we can lock in the metadata.
            # The message_start event is only sent when the first actual content arrives,
            # so this update will happen before that.
            if chunk.responseId:
                self.streamer.meta_data_captured = True

        yield from self.streamer.format_chunk(chunk)

    def format_error_chunk(self, message: str, status_code: int = 500) -> str:
        """Formats an error message using the Claude-specific 'error' event type."""
        # Claude uses a specific event type for errors
        error_payload = {"error": {"type": "server_error", "message": message}}
        return f"event: error\ndata: {json.dumps(error_payload)}\n\n"

    def format_response(
        self,
        response: GeminiResponse,
        original_request: Any,
    ) -> BaseModel:
        return gemini_response_to_claude(response, original_request)

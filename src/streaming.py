import json
from typing import Any, AsyncGenerator, Callable, Dict, List

import httpx
from pydantic import ValidationError

from .logger import format_log, get_logger
from .models import (
    GeminiResponse,
    GeminiContent,
    GeminiPart,
    GeminiCandidate,
    OpenAIChatCompletionStreamResponse,
)
from .openai_transformers import gemini_stream_chunk_to_openai
from .settings import settings

logger = get_logger(__name__)


class StreamParseError(Exception):
    """Custom exception for errors during stream parsing."""

    def __init__(self, message: str, original_text: str):
        super().__init__(message)
        self.original_text = original_text


# --- Core Components: Parsers, Transformers, Formatters ---


async def _parse_google_sse(
    response: httpx.Response,
) -> AsyncGenerator[GeminiResponse, None]:
    """Base parser that handles SSE protocol and yields raw GeminiResponse objects."""
    async for line in response.aiter_lines():
        if not line.startswith("data: "):
            continue
        data_str = line[len("data: ") :].strip()
        if not data_str:
            continue

        try:
            api_response_obj = json.loads(data_str)
            if "response" not in api_response_obj:
                continue
            yield GeminiResponse.model_validate(api_response_obj["response"])
        except (json.JSONDecodeError, ValidationError) as e:
            raise StreamParseError(f"Validation/JSON error: {e}", data_str)


def _coalesce_and_wrap_thoughts(
    parts: List[GeminiPart], tags: List[str]
) -> List[GeminiPart]:
    """
    Coalesces consecutive thought parts and wraps them in tags.
    """
    if not parts:
        return []

    start_tag, end_tag = tags
    new_parts: List[GeminiPart] = []
    thought_buffer: List[str] = []

    for part in parts:
        is_thought = part.thought and part.text
        if is_thought:
            thought_buffer.append(part.text)
        else:
            if thought_buffer:
                full_thought = "".join(thought_buffer)
                wrapped_thought = f"{start_tag}{full_thought}{end_tag}"
                new_parts.append(GeminiPart(text=wrapped_thought, thought=True))
                thought_buffer = []
            new_parts.append(part)

    if thought_buffer:
        full_thought = "".join(thought_buffer)
        wrapped_thought = f"{start_tag}{full_thought}{end_tag}"
        new_parts.append(GeminiPart(text=wrapped_thought, thought=True))

    return new_parts


async def _transform_wrap_thoughts(
    stream: AsyncGenerator[GeminiResponse, None], tags: List[str]
) -> AsyncGenerator[GeminiResponse, None]:
    """
    Transforms a stream to coalesce and wrap thought parts in the provided tags.
    This version is stateful to handle thoughts spanning multiple stream chunks.
    """
    start_tag, end_tag = tags
    thought_buffer = []

    async for chunk in stream:
        # This will hold the parts we want to yield in the current iteration
        parts_to_yield = []

        # Process all parts in the incoming chunk, assuming one candidate
        if not chunk.candidates or not chunk.candidates[0].content:
            yield chunk
            continue

        for part in chunk.candidates[0].content.parts:
            is_thought = hasattr(part, "thought") and part.thought and part.text
            if is_thought:
                # It's a thought, add it to our main buffer
                thought_buffer.append(part.text)
            else:
                # It's not a thought. This means any buffered thoughts are now complete.
                if thought_buffer:
                    # Flush the buffer into a single wrapped part
                    full_thought = "".join(thought_buffer)
                    wrapped_thought = f"{start_tag}{full_thought}{end_tag}"
                    parts_to_yield.append(
                        GeminiPart(text=wrapped_thought, thought=True)
                    )
                    thought_buffer = []  # Reset buffer

                # Add the non-thought part to be yielded
                parts_to_yield.append(part)

        # If we have parts to yield (because a non-thought was encountered),
        # create and yield a new chunk.
        if parts_to_yield:
            new_content = chunk.candidates[0].content.model_copy(
                update={"parts": parts_to_yield}
            )
            new_candidate = chunk.candidates[0].model_copy(
                update={"content": new_content}
            )
            yield chunk.model_copy(update={"candidates": [new_candidate]})

    # After the stream is finished, if there's anything left in the buffer,
    # it must be a block of thoughts at the very end. Flush it.
    if thought_buffer:
        full_thought = "".join(thought_buffer)
        wrapped_thought = f"{start_tag}{full_thought}{end_tag}"
        final_part = GeminiPart(text=wrapped_thought, thought=True)
        final_content = GeminiContent(role="model", parts=[final_part])
        # We need a candidate to wrap this in. Let's create a minimal one.
        final_candidate = GeminiCandidate(content=final_content, index=0)
        yield GeminiResponse(candidates=[final_candidate])


def wrap_thoughts_in_gemini_response(
    response: GeminiResponse, tags: List[str]
) -> GeminiResponse:
    """Transforms a GeminiResponse to wrap thought parts in the provided tags."""
    new_candidates = []
    for candidate in response.candidates:
        transformed_parts = _coalesce_and_wrap_thoughts(candidate.content.parts, tags)
        new_content = candidate.content.model_copy(update={"parts": transformed_parts})
        new_candidates.append(candidate.model_copy(update={"content": new_content}))
    return response.model_copy(update={"candidates": new_candidates})


def format_as_gemini_sse(chunk: GeminiResponse, **kwargs) -> str:
    """Formats a chunk into a Gemini-native SSE string."""
    return f"data: {chunk.model_dump_json(exclude_unset=True)}\n\n"


def format_as_openai_sse(chunk: GeminiResponse, **context) -> str:
    """
    Formats a GeminiResponse chunk into an OpenAI-compatible SSE string.
    This function now consistently uses the gemini_stream_chunk_to_openai
    transformer for all cases to ensure consistency and simplify maintenance,
    as recommended by code review.
    """
    response_id = context["response_id"]
    model = context["model"]

    # The transformer returns a dict; we must validate it into a Pydantic model
    # before we can serialize it. This aligns with the non-streaming logic.
    openai_chunk_dict = gemini_stream_chunk_to_openai(chunk, model, response_id)
    openai_chunk = OpenAIChatCompletionStreamResponse.model_validate(openai_chunk_dict)
    return f"data: {openai_chunk.model_dump_json(exclude_unset=True)}\n\n"


# --- Pipeline Orchestrator ---


async def process_stream_for_client(
    response: httpx.Response,
    formatter: Callable[..., str],
    formatter_context: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """The main pipeline orchestrator for processing and formatting streams."""
    try:
        pipeline = _parse_google_sse(response)

        # Only apply thought wrapping if the tags are properly configured
        if settings.THOUGHT_WRAPPER_TAGS and len(settings.THOUGHT_WRAPPER_TAGS) == 2:
            pipeline = _transform_wrap_thoughts(pipeline, settings.THOUGHT_WRAPPER_TAGS)

        # Word-by-word transformation can be enabled here if desired
        # pipeline = _transform_word_by_word(pipeline)

        full_response_text_for_log = []
        async for chunk in pipeline:
            if settings.DEBUG:
                # Extract text content from the chunk for logging
                text_content = "".join(
                    p.text for c in chunk.candidates for p in c.content.parts if p.text
                )
                if text_content:
                    full_response_text_for_log.append(text_content)

            yield formatter(chunk, **formatter_context)

        if settings.DEBUG and full_response_text_for_log:
            final_log_text = "".join(full_response_text_for_log)
            logger.debug(
                format_log("Full Stream Content Sent to Client", final_log_text)
            )

    except StreamParseError as e:
        logger.error(f"Error parsing stream from Google: {e.original_text}")
        error_payload = {"error": {"message": f"Stream parsing error: {e}"}}
        yield f"data: {json.dumps(error_payload)}\n\n"
    except Exception as e:
        logger.error(f"Generic stream processing error: {e}", exc_info=True)
        error_payload = {"error": {"message": f"An unexpected error occurred during streaming: {e}"}}
        yield f"data: {json.dumps(error_payload)}\n\n"

    if formatter_context.get("is_openai", False):
        yield "data: [DONE]\n\n"

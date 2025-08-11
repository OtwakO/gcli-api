import json
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from pydantic import BaseModel

from ..models.claude import (
    ClaudeContentBlock,
    ClaudeContentBlockDelta,
    ClaudeContentBlockStart,
    ClaudeContentBlockStop,
    ClaudeInputJsonDelta,
    ClaudeMessageDelta,
    ClaudeMessageResponse,
    ClaudeMessagesRequest,
    ClaudeMessageStartEvent,
    ClaudeMessageStop,
    ClaudeTextDelta,
    ClaudeUsage,
)
from ..models.gemini import (
    GeminiContent,
    GeminiPart,
    GeminiRequest,
    GeminiResponse,
    GeminiSystemInstruction,
)
from ..utils.logger import get_logger
from ..utils.utils import (
    generate_response_id,
    sanitize_gemini_tools,
)

logger = get_logger(__name__)


def _map_gemini_to_claude_finish_reason(gemini_reason: Optional[str]) -> str:
    """Maps Gemini's finish reason to Claude's, with logging for unexpected cases."""
    if not gemini_reason:
        logger.warning(
            "Upstream response candidate missing finishReason, defaulting to 'end_turn'."
        )
        return "end_turn"

    reason_map = {
        "STOP": "end_turn",
        "MAX_TOKENS": "max_tokens",
        "TOOL_USE": "tool_use",
    }

    if gemini_reason in reason_map:
        return reason_map[gemini_reason]

    # Handle other reasons like SAFETY, RECITATION, etc.
    logger.warning(
        f"Received unhandled Gemini finishReason '{gemini_reason}', defaulting to 'stop'."
    )
    return "stop"  # A generic fallback for other cases


def _transform_claude_content(
    content: Union[str, List[Dict[str, Any]]],
) -> List[GeminiPart]:
    """Transforms a Claude content block into a list of Gemini parts."""
    parts = []
    if isinstance(content, str):
        parts.append(GeminiPart(text=content))
        return parts

    for part_data in content:
        part_type = part_data.get("type")
        if part_type == "text":
            parts.append(GeminiPart(text=part_data.get("text", "")))
        elif part_type == "tool_use":
            parts.append(
                GeminiPart(
                    functionCall={
                        "name": part_data.get("name"),
                        "args": part_data.get("input", {}),
                    }
                )
            )
        elif part_type == "tool_result":
            parts.append(
                GeminiPart(
                    functionResponse={
                        "name": part_data.get("tool_use_id"),
                        "response": {"content": part_data.get("content")},
                    }
                )
            )
    return parts


def claude_request_to_gemini(
    claude_request: ClaudeMessagesRequest,
) -> Tuple[str, GeminiRequest]:
    """
    Transforms a Claude-compatible request payload into a Gemini-compatible Pydantic model.
    """
    model_name = claude_request.model

    contents = []
    for message in claude_request.messages:
        is_tool_response = isinstance(message.content, list) and any(
            p.get("type") == "tool_result" for p in message.content
        )
        role = (
            "tool"
            if is_tool_response
            else ("model" if message.role == "assistant" else "user")
        )
        parts = _transform_claude_content(message.content)
        if parts:
            contents.append(GeminiContent(role=role, parts=parts))

    system_instruction = None
    if claude_request.system:
        system_text = (
            claude_request.system
            if isinstance(claude_request.system, str)
            else json.dumps(claude_request.system)
        )
        system_instruction = GeminiSystemInstruction(
            parts=[GeminiPart(text=system_text)]
        )

    generation_config = {
        "maxOutputTokens": claude_request.max_tokens,
        "temperature": claude_request.temperature,
        "topP": claude_request.top_p,
        "topK": claude_request.top_k,
        "stopSequences": claude_request.stop_sequences,
    }
    generation_config = {k: v for k, v in generation_config.items() if v is not None}

    if (
        claude_request.response_format
        and claude_request.response_format.get("type") == "json_object"
    ):
        generation_config["responseMimeType"] = "application/json"

    tools = None
    if claude_request.tools:
        gemini_tools = [
            {
                "name": tool.get("name"),
                "description": tool.get("description"),
                "parameters": tool.get("input_schema"),
            }
            for tool in claude_request.tools
        ]
        tools = [{"functionDeclarations": gemini_tools}]

    sanitized_tools = sanitize_gemini_tools(tools)

    gemini_request = GeminiRequest(
        contents=contents,
        systemInstruction=system_instruction,
        generationConfig=generation_config,
        tools=sanitized_tools,
    )

    return model_name, gemini_request


class ClaudeStreamer:
    """Manages the state for transforming a Gemini stream to the Claude SSE format."""

    def __init__(self, response_id: str, model: str):
        self.response_id = response_id
        self.model = model
        self.is_finished = False
        self.content_block_index = 0
        self.last_block_type = None
        self.message_started = False
        self.meta_data_captured = False

    def _format_event(self, event_type: str, data: BaseModel) -> str:
        json_data = data.model_dump_json()
        return f"event: {event_type}\ndata: {json_data}\n\n"

    def _ensure_message_started(
        self,
        chunk: Optional[GeminiResponse],
    ) -> Generator[str, None, None]:
        """Yields the message_start event if it hasn't been sent yet."""
        if not self.message_started:
            self.message_started = True
            input_tokens = (
                chunk.usageMetadata.promptTokenCount or 0
                if chunk and chunk.usageMetadata
                else 0
            )
            message_response = ClaudeMessageResponse(
                id=self.response_id,
                model=self.model,
                content=[],
                usage=ClaudeUsage(input_tokens=input_tokens, output_tokens=0),
            )
            yield self._format_event(
                "message_start", ClaudeMessageStartEvent(message=message_response)
            )

    def format_chunk(
        self, chunk: Optional[GeminiResponse]
    ) -> Generator[str, None, None]:
        if self.is_finished:
            return

        has_content = chunk and chunk.candidates and chunk.candidates[0].content
        is_final_chunk = chunk is None or (
            chunk and chunk.candidates and chunk.candidates[0].finishReason
        )

        # --- Process Content ---
        if has_content:
            yield from self._ensure_message_started(chunk)

            for part in chunk.candidates[0].content.parts:
                current_block_type = None
                delta_model: Optional[Union[ClaudeTextDelta, ClaudeInputJsonDelta]] = (
                    None
                )
                start_block_model: Optional[ClaudeContentBlock] = None

                if part.text:
                    current_block_type = "text"
                    delta_model = ClaudeTextDelta(text=part.text)
                    start_block_model = ClaudeContentBlock(type="text", text="")
                elif part.functionCall:
                    current_block_type = "tool_use"
                    fc = part.functionCall
                    delta_model = ClaudeInputJsonDelta(
                        partial_json=json.dumps(fc.args, ensure_ascii=False)
                    )
                    start_block_model = ClaudeContentBlock(
                        type="tool_use", id=fc.name, name=fc.name, input={}
                    )

                if not current_block_type or not delta_model or not start_block_model:
                    continue

                if self.last_block_type and self.last_block_type != current_block_type:
                    yield self._format_event(
                        "content_block_stop",
                        ClaudeContentBlockStop(index=self.content_block_index),
                    )
                    self.content_block_index += 1
                    self.last_block_type = None

                if not self.last_block_type:
                    yield self._format_event(
                        "content_block_start",
                        ClaudeContentBlockStart(
                            index=self.content_block_index,
                            content_block=start_block_model,
                        ),
                    )
                    self.last_block_type = current_block_type

                yield self._format_event(
                    "content_block_delta",
                    ClaudeContentBlockDelta(
                        index=self.content_block_index, delta=delta_model
                    ),
                )

        # --- Handle End of Stream ---
        if is_final_chunk:
            self.is_finished = True
            yield from self._ensure_message_started(chunk)

            output_tokens = 0
            stop_reason = "end_turn"
            if chunk:
                if chunk.usageMetadata:
                    output_tokens = chunk.usageMetadata.candidatesTokenCount or 0
                if chunk.candidates:
                    stop_reason = _map_gemini_to_claude_finish_reason(
                        chunk.candidates[0].finishReason
                    )

            if self.last_block_type:
                # If the stream ends while a tool call is active, prioritize that as the stop reason.
                if self.last_block_type == "tool_use":
                    stop_reason = "tool_use"

                yield self._format_event(
                    "content_block_stop",
                    ClaudeContentBlockStop(index=self.content_block_index),
                )

            yield self._format_event(
                "message_delta",
                ClaudeMessageDelta(
                    delta={"stop_reason": stop_reason, "stop_sequence": None},
                    usage={"output_tokens": output_tokens},
                ),
            )
            yield self._format_event("message_stop", ClaudeMessageStop())
            return


def gemini_response_to_claude(
    gemini_response: GeminiResponse, original_request: ClaudeMessagesRequest
) -> ClaudeMessageResponse:
    """
    Transforms a non-streaming Gemini response into a Claude-compatible Pydantic model.
    """
    content_blocks = []
    stop_reason = None

    if gemini_response.candidates:
        candidate = gemini_response.candidates[0]
        stop_reason = _map_gemini_to_claude_finish_reason(candidate.finishReason)
        if candidate.content and candidate.content.parts:
            for part in candidate.content.parts:
                if part.text:
                    content_blocks.append(
                        ClaudeContentBlock(type="text", text=part.text)
                    )
                elif part.functionCall:
                    fc = part.functionCall
                    content_blocks.append(
                        ClaudeContentBlock(
                            type="tool_use", id=fc.name, name=fc.name, input=fc.args
                        )
                    )

    input_tokens = 0
    output_tokens = 0
    if gemini_response.usageMetadata:
        input_tokens = gemini_response.usageMetadata.promptTokenCount or 0
        output_tokens = gemini_response.usageMetadata.candidatesTokenCount or 0

    # Use upstream response data when available, with robust fallbacks
    if gemini_response.responseId:
        response_id = f"msg_{gemini_response.responseId}"
    elif original_request.id:
        response_id = f"msg_{original_request.id}"
    else:
        response_id = generate_response_id("msg")

    model = gemini_response.modelVersion or original_request.model

    return ClaudeMessageResponse(
        id=response_id,
        model=model,
        content=content_blocks,
        stop_reason=stop_reason,
        usage=ClaudeUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )

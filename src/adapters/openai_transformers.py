import json
import time
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

from ..models.gemini import (
    BatchEmbedContentResponse,
    EmbedContentResponse,
    GeminiContent,
    GeminiPart,
    GeminiRequest,
    GeminiResponse,
    GeminiSystemInstruction,
)
from ..models.openai import (
    FunctionCall,
    OpenAIChatCompletionChoice,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIChatCompletionStreamChoice,
    OpenAIChatCompletionStreamResponse,
    OpenAIChatMessage,
    OpenAIDelta,
    OpenAIEmbeddingData,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
    OpenAIUsage,
    ToolCall,
)
from ..utils.constants import DEFAULT_SAFETY_SETTINGS
from ..utils.logger import get_logger
from ..utils.utils import (
    generate_response_id,
    sanitize_gemini_tools,
)

logger = get_logger(__name__)


def openai_embedding_request_transformer(
    req: OpenAIEmbeddingRequest,
) -> Tuple[str, str, Dict[str, Any]]:
    """
    Transforms an OpenAI Embedding request into the appropriate Gemini format,
    handling both single and batch requests.

    Returns a tuple containing the determined action, the model name, and the
    request body for the Gemini API.
    """
    model_name = req.model

    # Handle batch requests
    if isinstance(req.input, list):
        logger.info(f"Handling batch embedding request for {len(req.input)} items.")
        action = "batchEmbedContents"
        gemini_requests = []
        for text_input in req.input:
            # Basic validation, skipping non-string inputs in the list
            if isinstance(text_input, str):
                gemini_requests.append(
                    {
                        "model": model_name,
                        "content": {"parts": [{"text": text_input}]},
                    }
                )
            else:
                logger.warning(
                    f"Skipping non-string item in batch embedding input: {type(text_input)}"
                )
        request_body = {"requests": gemini_requests}
        return action, req.model, request_body

    # Handle single string request
    elif isinstance(req.input, str):
        logger.info("Handling single embedding request.")
        action = "embedContent"
        request_body = {"content": {"parts": [{"text": req.input}]}}
        return action, req.model, request_body

    # Handle unsupported types
    else:
        logger.error(f"Unsupported embedding input type: {type(req.input)}")
        raise TypeError("Unsupported embedding input type. Must be str or list[str].")


def gemini_response_to_openai_embedding(
    gemini_response: Union[EmbedContentResponse, BatchEmbedContentResponse],
    original_request: OpenAIEmbeddingRequest,
) -> OpenAIEmbeddingResponse:
    """Transforms a Gemini embedding response into an OpenAI-compatible one."""
    embedding_data_list = []

    if isinstance(gemini_response, EmbedContentResponse):
        # Handle single embedding response
        embedding_data_list.append(
            OpenAIEmbeddingData(embedding=gemini_response.embedding.values, index=0)
        )
    elif isinstance(gemini_response, BatchEmbedContentResponse):
        # Handle batch embedding response
        for i, embedding in enumerate(gemini_response.embeddings):
            embedding_data_list.append(
                OpenAIEmbeddingData(embedding=embedding.values, index=i)
            )

    # Placeholder for token count, as the Gemini SDK doesn't directly provide it
    # in the embedding response. A separate countTokens call would be needed for accuracy.
    usage = OpenAIUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    return OpenAIEmbeddingResponse(
        data=embedding_data_list,
        model=original_request.model,
        usage=usage,
    )


def _transform_message_part(
    part: Dict[str, Any], message_index: int
) -> Optional[GeminiPart]:
    """Transforms a single part of an OpenAI message content into a GeminiPart."""
    part_type = part.get("type")
    if part_type == "text":
        return GeminiPart(text=part.get("text", ""))
    elif part_type == "image_url":
        image_url = part.get("image_url", {}).get("url")
        if not image_url or ";base64," not in image_url:
            logger.warning(
                f"Skipping invalid or non-base64 image_url part in message {message_index}: {part}"
            )
            return None
        try:
            header, base64_data = image_url.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            return GeminiPart(inlineData={"mimeType": mime_type, "data": base64_data})
        except (ValueError, IndexError) as e:
            logger.warning(
                f"Could not parse image_url in message {message_index}: {image_url}. Error: {e}"
            )
            return None
    return None


def _transform_messages(messages: List[OpenAIChatMessage]) -> List[GeminiContent]:
    """Transforms a list of OpenAI messages to a list of GeminiContent objects."""
    contents = []
    for i, message in enumerate(messages):
        role = (
            "model"
            if message.role == "assistant"
            else "tool"
            if message.role == "tool"
            else "user"
        )

        parts = []
        if role == "tool":
            parts.append(
                GeminiPart(
                    functionResponse={
                        "name": message.tool_call_id,
                        "response": {"content": message.content},
                    }
                )
            )
        elif isinstance(message.content, list):
            for part_data in message.content:
                transformed_part = _transform_message_part(part_data, i)
                if transformed_part:
                    parts.append(transformed_part)
        elif message.content is not None:
            parts.append(GeminiPart(text=message.content))

        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Could not decode tool call arguments: {tool_call.function.arguments}"
                    )
                    args = {}
                parts.append(
                    GeminiPart(
                        functionCall={
                            "name": tool_call.function.name,
                            "args": args,
                        }
                    )
                )

        if parts:
            contents.append(GeminiContent(role=role, parts=parts))

    return contents


def _transform_generation_config(req: OpenAIChatCompletionRequest) -> Dict[str, Any]:
    """Builds the generationConfig dictionary from an OpenAI request."""
    config = {
        "temperature": req.temperature,
        "topP": req.top_p,
        "maxOutputTokens": req.max_tokens,
        "stopSequences": [req.stop] if isinstance(req.stop, str) else req.stop,
        "frequencyPenalty": req.frequency_penalty,
        "presencePenalty": req.presence_penalty,
        "candidateCount": req.n,
        "seed": req.seed,
    }
    if req.response_format and req.response_format.get("type") == "json_object":
        config["responseMimeType"] = "application/json"

    return {k: v for k, v in config.items() if v is not None}


def _transform_tools(
    req: OpenAIChatCompletionRequest,
) -> Optional[List[Dict[str, Any]]]:
    """Builds the tools list from an OpenAI request."""
    if not req.tools:
        return None
    return [{"functionDeclarations": [t["function"] for t in req.tools]}]


def _transform_tool_config(
    req: OpenAIChatCompletionRequest,
) -> Optional[Dict[str, Any]]:
    """Builds the toolConfig dictionary from an OpenAI request."""
    if not req.tool_choice:
        return None

    mode = None
    allowed_function_names = None

    if isinstance(req.tool_choice, str):
        if req.tool_choice == "none":
            mode = "NONE"
        elif req.tool_choice == "auto":
            mode = "AUTO"
        elif req.tool_choice == "required":
            # "required" in OpenAI means the model must call a tool.
            # The closest Gemini equivalent is "ANY", which forces a call from the available tools.
            mode = "ANY"
    elif isinstance(req.tool_choice, dict):
        function_name = req.tool_choice.get("function", {}).get("name")
        if function_name:
            mode = "ANY"
            allowed_function_names = [function_name]

    if mode:
        config = {"mode": mode}
        if allowed_function_names:
            config["allowedFunctionNames"] = allowed_function_names
        return {"functionCallingConfig": config}

    return None


def openai_request_to_gemini(req: OpenAIChatCompletionRequest) -> GeminiRequest:
    """Converts an OpenAI Chat Completion request to a GeminiRequest Pydantic model."""
    system_instruction = None
    messages = list(req.messages)

    system_message_index = -1
    for i, msg in enumerate(messages):
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_instruction = GeminiSystemInstruction(
                    parts=[GeminiPart(text=msg.content)]
                )
            system_message_index = i
            break

    if system_message_index != -1:
        messages.pop(system_message_index)

    generation_config = _transform_generation_config(req)

    # Transform and then sanitize the tools
    transformed_tools = _transform_tools(req)
    sanitized_tools = sanitize_gemini_tools(transformed_tools)

    gemini_request = GeminiRequest(
        contents=_transform_messages(messages),
        generationConfig=generation_config,
        safetySettings=DEFAULT_SAFETY_SETTINGS,
        systemInstruction=system_instruction,
        tools=sanitized_tools,
        toolConfig=_transform_tool_config(req),
    )

    return gemini_request


def _gemini_candidate_to_openai_choices(
    candidate,
    is_streaming: bool = False,
) -> Generator[
    Union[OpenAIChatCompletionStreamChoice, OpenAIChatCompletionChoice], None, None
]:
    """
    Transforms a single Gemini candidate into a generator of one or more
    OpenAI choice objects, handling mixed text and tool call content.
    """
    parts = candidate.content.parts
    total_parts = len(parts)

    for i, part in enumerate(parts):
        is_last_part = i == total_parts - 1
        finish_reason = (
            _map_finish_reason(candidate.finishReason, is_streaming=is_streaming)
            if is_last_part
            else None
        )

        # Part 1: Yield a text choice if text exists
        if part.text:
            if is_streaming:
                delta = OpenAIDelta(content=part.text)
                # Only the first part from a mixed content part should have the role
                if not part.functionCall:
                    delta.role = "assistant"
                yield OpenAIChatCompletionStreamChoice(
                    index=candidate.index,
                    delta=delta,
                    # Finish reason is only sent with the very last yielded choice
                    finish_reason=finish_reason if not part.functionCall else None,
                )
            else:
                message = OpenAIChatMessage(role="assistant", content=part.text)
                yield OpenAIChatCompletionChoice(
                    index=candidate.index, message=message, finish_reason=finish_reason
                )

        # Part 2: Yield a tool call choice if a function call exists
        if part.functionCall:
            fc = part.functionCall
            tool_call = ToolCall(
                id=fc.name,
                function=FunctionCall(name=fc.name, arguments=json.dumps(fc.args)),
            )

            if is_streaming:
                tool_call.index = 0  # Add index for streaming tool calls
                delta = OpenAIDelta(tool_calls=[tool_call], role="assistant")
                yield OpenAIChatCompletionStreamChoice(
                    index=candidate.index, delta=delta, finish_reason=finish_reason
                )
            else:
                # For non-streaming, the message content should be None when there are tool calls
                message = OpenAIChatMessage(
                    role="assistant", content=None, tool_calls=[tool_call]
                )
                yield OpenAIChatCompletionChoice(
                    index=candidate.index, message=message, finish_reason=finish_reason
                )


def gemini_response_to_openai(
    gemini_response: GeminiResponse, original_request: OpenAIChatCompletionRequest
) -> OpenAIChatCompletionResponse:
    """Transforms a Gemini response into an OpenAI-compatible one."""
    choices = []
    for c in gemini_response.candidates:
        choices.extend(list(_gemini_candidate_to_openai_choices(c, is_streaming=False)))

    usage = OpenAIUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    if gemini_response.usageMetadata:
        usage.prompt_tokens = gemini_response.usageMetadata.promptTokenCount or 0
        usage.completion_tokens = (
            gemini_response.usageMetadata.candidatesTokenCount or 0
        )
        usage.total_tokens = gemini_response.usageMetadata.totalTokenCount or 0

    response_id = (
        f"chatcmpl-{gemini_response.responseId}"
        if gemini_response.responseId
        else generate_response_id("chatcmpl")
    )
    model = gemini_response.modelVersion or original_request.model

    created_timestamp = int(time.time())
    if gemini_response.createTime:
        try:
            dt_object = datetime.fromisoformat(
                gemini_response.createTime.replace("Z", "+00:00")
            )
            created_timestamp = int(dt_object.timestamp())
        except (ValueError, TypeError):
            logger.warning(
                f"Could not parse createTime '{gemini_response.createTime}', falling back to current time."
            )

    return OpenAIChatCompletionResponse(
        id=response_id,
        object="chat.completion",
        created=created_timestamp,
        model=model,
        choices=choices,
        usage=usage,
    )


def gemini_stream_chunk_to_openai(
    gemini_chunk: GeminiResponse, model: str, response_id: str
) -> OpenAIChatCompletionStreamResponse:
    """Builds an OpenAI-compatible stream chunk from a Gemini response chunk."""
    choices = []
    for c in gemini_chunk.candidates:
        choices.extend(list(_gemini_candidate_to_openai_choices(c, is_streaming=True)))

    return OpenAIChatCompletionStreamResponse(
        id=response_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=choices,
    )


def _map_finish_reason(
    gemini_reason: Optional[str], is_streaming: bool = False
) -> Optional[str]:
    """
    Maps Gemini's finish reason to OpenAI's, handling streaming context.
    For intermediate streaming chunks, a missing reason is expected and returns None.
    For non-streaming responses, a missing reason is logged as a warning.
    """
    if not gemini_reason:
        if is_streaming:
            return None  # Expected for intermediate stream chunks
        else:
            logger.warning(
                "Non-streaming upstream response candidate missing finishReason, defaulting to 'stop'."
            )
            return "stop"

    reason_map = {
        "STOP": "stop",
        "MAX_TOKENS": "length",
        "SAFETY": "content_filter",
        "RECITATION": "content_filter",
        "TOOL_USE": "tool_calls",
    }

    if gemini_reason in reason_map:
        return reason_map[gemini_reason]

    logger.warning(
        f"Received unhandled Gemini finishReason '{gemini_reason}', defaulting to 'stop'."
    )
    return "stop"

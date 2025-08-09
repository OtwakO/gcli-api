import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union, Tuple

from ..utils.constants import DEFAULT_SAFETY_SETTINGS
from ..utils.logger import get_logger
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
    FunctionCall,
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

    gemini_request = GeminiRequest(
        contents=_transform_messages(messages),
        generationConfig=_transform_generation_config(req),
        safetySettings=DEFAULT_SAFETY_SETTINGS,
        systemInstruction=system_instruction,
        tools=_transform_tools(req),
        toolConfig=_transform_tool_config(req),
    )

    return gemini_request


def _gemini_candidate_to_openai_choice(
    candidate, is_streaming: bool = False
) -> Union[OpenAIChatCompletionStreamChoice, OpenAIChatCompletionChoice]:
    """Transforms a single Gemini candidate into an OpenAI choice object."""
    parts = candidate.content.parts
    tool_calls = []
    content_text = None

    for part in parts:
        if part.functionCall:
            fc = part.functionCall
            tool_call = ToolCall(
                id=fc.name,
                function=FunctionCall(name=fc.name, arguments=json.dumps(fc.args)),
            )
            if is_streaming:
                tool_call.index = 0  # Add index for streaming tool calls
            tool_calls.append(tool_call)
        if part.text:
            content_text = part.text

    if is_streaming:
        delta = OpenAIDelta()
        if content_text:
            delta.content = content_text
        if tool_calls:
            delta.tool_calls = tool_calls
            delta.role = "assistant"
        return OpenAIChatCompletionStreamChoice(
            index=candidate.index,
            delta=delta,
            finish_reason=_map_finish_reason(candidate.finishReason),
        )
    else:
        message = OpenAIChatMessage(role="assistant", content=content_text)
        if tool_calls:
            message.tool_calls = tool_calls
            message.content = None
        return OpenAIChatCompletionChoice(
            index=candidate.index,
            message=message,
            finish_reason=_map_finish_reason(candidate.finishReason),
        )


def gemini_response_to_openai(
    gemini_response: GeminiResponse, original_request: OpenAIChatCompletionRequest
) -> OpenAIChatCompletionResponse:
    """Transforms a Gemini response into an OpenAI-compatible one."""
    choices = [
        _gemini_candidate_to_openai_choice(c, is_streaming=False)
        for c in gemini_response.candidates
    ]

    usage = OpenAIUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    if gemini_response.usageMetadata:
        usage.prompt_tokens = gemini_response.usageMetadata.promptTokenCount or 0
        usage.completion_tokens = (
            gemini_response.usageMetadata.candidatesTokenCount or 0
        )
        usage.total_tokens = gemini_response.usageMetadata.totalTokenCount or 0

    return OpenAIChatCompletionResponse(
        id="chatcmpl-" + str(uuid.uuid4()),
        object="chat.completion",
        created=int(time.time()),
        model=original_request.model,
        choices=choices,
        usage=usage,
    )


def gemini_stream_chunk_to_openai(
    gemini_chunk: GeminiResponse, model: str, response_id: str
) -> OpenAIChatCompletionStreamResponse:
    """Builds an OpenAI-compatible stream chunk from a Gemini response chunk."""
    choices = [
        _gemini_candidate_to_openai_choice(c, is_streaming=True)
        for c in gemini_chunk.candidates
    ]
    return OpenAIChatCompletionStreamResponse(
        id=response_id,
        object="chat.completion.chunk",
        created=int(time.time()),
        model=model,
        choices=choices,
    )


def _map_finish_reason(gemini_reason: Optional[str]) -> str:
    """Maps Gemini's finish reason to OpenAI's, with logging for unexpected cases."""
    if not gemini_reason:
        logger.warning(
            "Upstream response candidate missing finishReason, defaulting to 'stop'."
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

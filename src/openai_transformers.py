import json
import time
import uuid
from typing import Any, Dict, List, Optional

from .constants import DEFAULT_SAFETY_SETTINGS
from .models import (
    OpenAIChatCompletionRequest,
    GeminiResponse,
    OpenAIChatMessage,
    OpenAIEmbeddingRequest,
    EmbedContentResponse,
    OpenAIEmbeddingResponse,
    OpenAIEmbeddingData,
    OpenAIUsage,
)
from .logger import get_logger

logger = get_logger(__name__)


def openai_embedding_request_to_gemini(
    req: OpenAIEmbeddingRequest,
) -> Dict[str, Any]:
    """Converts an OpenAI Embedding request to a Gemini API request payload."""
    # The Gemini API expects a single content object.
    # If the input is a list, we can only process the first element for a single embedding.
    # This is a simplification; a more complex implementation could batch requests.
    text_input = ""
    if isinstance(req.input, str):
        text_input = req.input
    elif isinstance(req.input, list) and len(req.input) > 0:
        # For simplicity, we'll take the first item if it's a list of strings.
        # The Gemini API's embedContent takes a single content blob.
        if isinstance(req.input[0], str):
            text_input = req.input[0]
        else:
            # Handling for token arrays is not directly supported in this simplified flow.
            logger.warning("List of token arrays as input is not supported.")

    return {"content": {"parts": [{"text": text_input}]}}


def gemini_embedding_response_to_openai(
    gemini_response: EmbedContentResponse, model: str
) -> OpenAIEmbeddingResponse:
    """Transforms a Gemini embedding response into an OpenAI-compatible one."""
    embedding_data = OpenAIEmbeddingData(
        embedding=gemini_response.embedding.values, index=0
    )

    # Placeholder for token count, as the Gemini SDK doesn't directly provide it
    # in the embedding response. A separate countTokens call would be needed for accuracy.
    usage = OpenAIUsage(prompt_tokens=0, total_tokens=0)

    return OpenAIEmbeddingResponse(
        data=[embedding_data],
        model=model,
        usage=usage,
    )


def _transform_message_part(part: Dict[str, Any], message_index: int) -> Optional[Dict[str, Any]]:
    """Transforms a single part of an OpenAI message content."""
    part_type = part.get("type")
    if part_type == "text":
        return {"text": part.get("text", "")}
    elif part_type == "image_url":
        image_url = part.get("image_url", {}).get("url")
        if not image_url or ";base64," not in image_url:
            logger.warning(f"Skipping invalid or non-base64 image_url part in message {message_index}: {part}")
            return None
        try:
            # Format: data:image/jpeg;base64,LzlqLzRBQ...
            header, base64_data = image_url.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            return {"inlineData": {"mimeType": mime_type, "data": base64_data}}
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse image_url in message {message_index}: {image_url}. Error: {e}")
            return None
    return None

def _transform_messages(messages: List[OpenAIChatMessage]) -> List[Dict[str, Any]]:
    """Transforms a list of OpenAI messages to Gemini's content format."""
    contents = []
    for i, message in enumerate(messages):
        role = "model" if message.role == "assistant" else "tool" if message.role == "tool" else "user"
        
        parts = []
        if role == "tool":
            parts.append({
                "functionResponse": {
                    "name": message.tool_call_id,
                    "response": {"content": message.content},
                }
            })
        elif isinstance(message.content, list):
            for part in message.content:
                transformed_part = _transform_message_part(part, i)
                if transformed_part:
                    parts.append(transformed_part)
        elif message.content is not None:
            parts.append({"text": message.content})

        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    logger.warning(f"Could not decode tool call arguments: {tool_call.function.arguments}")
                    args = {}
                parts.append({
                    "functionCall": {
                        "name": tool_call.function.name,
                        "args": args,
                    }
                })
        
        if parts:
            contents.append({"role": role, "parts": parts})
            
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

def _transform_tools(req: OpenAIChatCompletionRequest) -> Optional[List[Dict[str, Any]]]:
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

def openai_request_to_gemini(req: OpenAIChatCompletionRequest) -> Dict[str, Any]:
    """Converts an OpenAI Chat Completion request to a Gemini API request payload."""
    system_instruction = None
    messages = list(req.messages)  # Make a copy to modify

    # Find and extract the system message if it exists
    system_message_index = -1
    for i, msg in enumerate(messages):
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_instruction = {"parts": [{"text": msg.content}]}
            system_message_index = i
            break

    # If a system message was found, remove it from the list
    if system_message_index != -1:
        messages.pop(system_message_index)

    payload = {
        "contents": _transform_messages(messages),
        "generationConfig": _transform_generation_config(req),
        "safetySettings": DEFAULT_SAFETY_SETTINGS,
    }

    if system_instruction:
        payload["systemInstruction"] = system_instruction

    tools = _transform_tools(req)
    if tools:
        payload["tools"] = tools

    tool_config = _transform_tool_config(req)
    if tool_config:
        payload["toolConfig"] = tool_config

    return payload


def _gemini_candidate_to_openai_choice(candidate, is_streaming: bool = False) -> Dict[str, Any]:
    """Transforms a single Gemini candidate into an OpenAI choice dictionary."""
    parts = candidate.content.parts
    tool_calls = []
    content_text = None

    for part in parts:
        if part.functionCall:
            fc = part.functionCall
            tool_call_data = {
                "id": fc.name,
                "type": "function",
                "function": {
                    "name": fc.name,
                    "arguments": json.dumps(fc.args),
                },
            }
            if is_streaming:
                tool_call_data["index"] = 0  # Add index for streaming tool calls
            tool_calls.append(tool_call_data)
        if part.text:
            content_text = part.text

    if is_streaming:
        delta = {}
        if content_text:
            delta["content"] = content_text
        if tool_calls:
            delta["tool_calls"] = tool_calls
            delta["role"] = "assistant"
        return {
            "index": candidate.index,
            "delta": delta,
            "finish_reason": _map_finish_reason(candidate.finish_reason),
        }
    else:
        message = {"role": "assistant", "content": content_text}
        if tool_calls:
            message["tool_calls"] = tool_calls
            message["content"] = None
        return {
            "index": candidate.index,
            "message": message,
            "finish_reason": _map_finish_reason(candidate.finish_reason),
        }

def gemini_response_to_openai(
    gemini_response: GeminiResponse, model: str
) -> Dict[str, Any]:
    choices = [
        _gemini_candidate_to_openai_choice(c, is_streaming=False)
        for c in gemini_response.candidates
    ]
    return {
        "id": "chatcmpl-" + str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }


def gemini_stream_chunk_to_openai(
    gemini_chunk: GeminiResponse, model: str, response_id: str
) -> Dict[str, Any]:
    choices = [
        _gemini_candidate_to_openai_choice(c, is_streaming=True)
        for c in gemini_chunk.candidates
    ]
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }


def _map_finish_reason(gemini_reason: str) -> str:
    if gemini_reason == "STOP":
        return "stop"
    if gemini_reason == "MAX_TOKENS":
        return "length"
    if gemini_reason in ["SAFETY", "RECITATION"]:
        return "content_filter"
    if gemini_reason == "TOOL_USE":
        return "tool_calls"
    return None


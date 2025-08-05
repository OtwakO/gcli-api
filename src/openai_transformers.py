import json
import time
import uuid
from typing import Any, Dict

from .constants import DEFAULT_SAFETY_SETTINGS
from .models import OpenAIChatCompletionRequest, GeminiResponse


import json
import time
import uuid
from typing import Any, Dict, List, Optional

from .constants import DEFAULT_SAFETY_SETTINGS
from .models import OpenAIChatCompletionRequest, GeminiResponse, OpenAIChatMessage
from .logger import get_logger

logger = get_logger(__name__)

def _transform_message_part(part: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Transforms a single part of an OpenAI message content."""
    part_type = part.get("type")
    if part_type == "text":
        return {"text": part.get("text", "")}
    elif part_type == "image_url":
        image_url = part.get("image_url", {}).get("url")
        if not image_url or ";base64," not in image_url:
            logger.warning(f"Skipping invalid or non-base64 image_url part: {part}")
            return None
        try:
            # Format: data:image/jpeg;base64,LzlqLzRBQ...            
            header, base64_data = image_url.split(",", 1)
            mime_type = header.split(":")[1].split(";")[0]
            return {"inlineData": {"mimeType": mime_type, "data": base64_data}}
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not parse image_url: {image_url}. Error: {e}")
            return None
    return None

def _transform_messages(messages: List[OpenAIChatMessage]) -> List[Dict[str, Any]]:
    """Transforms a list of OpenAI messages to Gemini's content format."""
    contents = []
    for message in messages:
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
                transformed_part = _transform_message_part(part)
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

def _transform_tool_config(req: OpenAIChatCompletionRequest) -> Optional[Dict[str, Any]]:
    """Builds the toolConfig dictionary from an OpenAI request."""
    if not req.tool_choice:
        return None
    
    if isinstance(req.tool_choice, str) and req.tool_choice in ["none", "auto"]:
        return {"functionCallingConfig": {"mode": req.tool_choice.upper()}}
    
    if isinstance(req.tool_choice, dict):
        function_name = req.tool_choice.get("function", {}).get("name")
        if function_name:
            return {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [function_name],
                }
            }
    return None

def openai_request_to_gemini(req: OpenAIChatCompletionRequest) -> Dict[str, Any]:
    """Converts an OpenAI Chat Completion request to a Gemini API request payload."""
    payload = {
        "contents": _transform_messages(req.messages),
        "generationConfig": _transform_generation_config(req),
        "safetySettings": DEFAULT_SAFETY_SETTINGS,  # Assuming constant for now
    }

    tools = _transform_tools(req)
    if tools:
        payload["tools"] = tools

    tool_config = _transform_tool_config(req)
    if tool_config:
        payload["toolConfig"] = tool_config

    return payload


def gemini_response_to_openai(
    gemini_response: GeminiResponse, model: str
) -> Dict[str, Any]:
    choices = []
    for candidate in gemini_response.candidates:
        parts = candidate.content.parts
        
        tool_calls = []
        content_text = None

        for part in parts:
            if part.functionCall:
                fc = part.functionCall
                tool_calls.append(
                    {
                        "id": fc.name,
                        "type": "function",
                        "function": {
                            "name": fc.name,
                            "arguments": json.dumps(fc.args),
                        },
                    }
                )
            if part.text:
                content_text = part.text

        message = {
            "role": "assistant",
            "content": content_text,
        }
        if tool_calls:
            message["tool_calls"] = tool_calls
            message["content"] = None

        choices.append(
            {
                "index": candidate.index,
                "message": message,
                "finish_reason": _map_finish_reason(candidate.finish_reason),
            }
        )
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
    choices = []
    for candidate in gemini_chunk.candidates:
        delta = {}
        parts = candidate.content.parts
        
        tool_calls = []
        content_text = None

        for part in parts:
            if part.text:
                content_text = part.text
            elif part.functionCall:
                fc = part.functionCall
                tool_call_chunk = {
                    "index": 0,
                    "id": fc.name,
                    "type": "function",
                    "function": {
                        "name": fc.name,
                        "arguments": json.dumps(fc.args),
                    },
                }
                tool_calls.append(tool_call_chunk)
        
        if content_text:
            delta["content"] = content_text
        
        if tool_calls:
            delta["tool_calls"] = tool_calls
            delta["role"] = "assistant"

        choices.append(
            {
                "index": candidate.index,
                "delta": delta,
                "finish_reason": _map_finish_reason(candidate.finish_reason),
            }
        )
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


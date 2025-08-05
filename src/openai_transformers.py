import json
import time
import uuid
from typing import Any, Dict

from .constants import DEFAULT_SAFETY_SETTINGS
from .models import OpenAIChatCompletionRequest, GeminiResponse


def openai_request_to_gemini(
    openai_request: OpenAIChatCompletionRequest,
) -> Dict[str, Any]:
    contents = []
    for message in openai_request.messages:
        role = "user"
        if message.role == "assistant":
            role = "model"
        elif message.role == "tool":
            role = "tool"

        parts = []
        if role == "tool":
            parts.append(
                {
                    "functionResponse": {
                        "name": message.tool_call_id,
                        "response": {"content": message.content},
                    }
                }
            )
        elif isinstance(message.content, list):
            for part in message.content:
                if part.get("type") == "text":
                    parts.append({"text": part.get("text", "")})
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url")
                    if image_url:
                        try:
                            mime_type, base64_data = image_url.split(";")
                            _, mime_type = mime_type.split(":")
                            _, base64_data = base64_data.split(",")
                            parts.append(
                                {
                                    "inlineData": {
                                        "mimeType": mime_type,
                                        "data": base64_data,
                                    }
                                }
                            )
                        except ValueError:
                            continue
        elif message.content is not None:
            parts.append({"text": message.content})

        if message.role == "assistant" and message.tool_calls:
            for tool_call in message.tool_calls:
                parts.append(
                    {
                        "functionCall": {
                            "name": tool_call.function.name,
                            "args": json.loads(tool_call.function.arguments),
                        }
                    }
                )
        
        contents.append({"role": role, "parts": parts})

    generation_config = {
        k: v
        for k, v in {
            "temperature": openai_request.temperature,
            "topP": openai_request.top_p,
            "maxOutputTokens": openai_request.max_tokens,
            "stopSequences": [openai_request.stop]
            if isinstance(openai_request.stop, str)
            else openai_request.stop,
            "frequencyPenalty": openai_request.frequency_penalty,
            "presencePenalty": openai_request.presence_penalty,
            "candidateCount": openai_request.n,
            "seed": openai_request.seed,
            "responseMimeType": "application/json"
            if openai_request.response_format
            and openai_request.response_format.get("type") == "json_object"
            else None,
        }.items()
        if v is not None
    }

    tools = None
    if openai_request.tools:
        tools = [{"functionDeclarations": [t["function"] for t in openai_request.tools]}]

    tool_config = None
    if openai_request.tool_choice:
        if isinstance(openai_request.tool_choice, str):
            if openai_request.tool_choice in ["none", "auto"]:
                tool_config = {
                    "functionCallingConfig": {"mode": openai_request.tool_choice.upper()}
                }
        elif isinstance(openai_request.tool_choice, dict):
            function_name = openai_request.tool_choice.get("function", {}).get("name")
            if function_name:
                tool_config = {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [function_name],
                    }
                }

    return {
        "contents": contents,
        "generationConfig": generation_config,
        "safetySettings": DEFAULT_SAFETY_SETTINGS,
        "model": openai_request.model,
        "tools": tools,
        "toolConfig": tool_config,
    }


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


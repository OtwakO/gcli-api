import json
import time
import uuid
from typing import Any, Dict

from .constants import DEFAULT_SAFETY_SETTINGS
from .models import OpenAIChatCompletionRequest


def openai_request_to_gemini(
    openai_request: OpenAIChatCompletionRequest,
) -> Dict[str, Any]:
    contents = []
    for message in openai_request.messages:
        role = "model" if message.role == "assistant" else "user"
        if isinstance(message.content, list):
            parts = []
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
            contents.append({"role": role, "parts": parts})
        else:
            contents.append({"role": role, "parts": [{"text": message.content}]})

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

    return {
        "contents": contents,
        "generationConfig": generation_config,
        "safetySettings": DEFAULT_SAFETY_SETTINGS,
        "model": openai_request.model,
    }


def gemini_response_to_openai(
    gemini_response: Dict[str, Any], model: str
) -> Dict[str, Any]:
    choices = []
    for candidate in gemini_response.get("candidates", []):
        role = (
            "assistant" if candidate.get("content", {}).get("role", "model") else "user"
        )
        parts = candidate.get("content", {}).get("parts", [])
        content = parts[0].get("text", "") if parts else ""
        choices.append(
            {
                "index": candidate.get("index", 0),
                "message": {"role": role, "content": content},
                "finish_reason": _map_finish_reason(candidate.get("finishReason")),
            }
        )
    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": choices,
    }


def gemini_stream_chunk_to_openai(
    gemini_chunk: Dict[str, Any], model: str, response_id: str
) -> Dict[str, Any]:
    choices = []
    for candidate in gemini_chunk.get("candidates", []):
        parts = candidate.get("content", {}).get("parts", [])
        content = parts[0].get("text", "") if parts else ""
        choices.append(
            {
                "index": candidate.get("index", 0),
                "delta": {"content": content},
                "finish_reason": _map_finish_reason(candidate.get("finishReason")),
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
    return None

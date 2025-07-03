import asyncio
import json
import logging

import httpx
from fastapi import Response
from fastapi.responses import StreamingResponse
from google.auth.transport.requests import Request as GoogleAuthRequest

from .auth import get_credentials, get_user_project_id, onboard_user, save_credentials
from .constants import CODE_ASSIST_ENDPOINT, DEFAULT_SAFETY_SETTINGS
from .settings import settings
from .utils import get_user_agent


async def send_gemini_request(payload: dict, is_streaming: bool = False) -> Response:
    creds = await get_credentials()
    if not creds:
        raise Exception("Proxy not authenticated. Please log in.")

    if creds.expired and creds.refresh_token:
        try:
            await asyncio.to_thread(creds.refresh, GoogleAuthRequest())
            save_credentials(creds)
        except Exception as e:
            raise Exception(f"Token refresh failed: {e}")

    proj_id = await get_user_project_id(creds)
    if not proj_id:
        raise Exception("Failed to get user project ID.")

    await onboard_user(creds, proj_id)

    action = "streamGenerateContent" if is_streaming else "generateContent"
    target_url = f"{CODE_ASSIST_ENDPOINT}/v1internal:{action}"
    if is_streaming:
        target_url += "?alt=sse"

    request_headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }

    final_payload = {
        "model": payload.get("model"),
        "project": proj_id,
        "request": payload.get("request", {}),
    }

    # # Always include thoughts in response.
    # if (request_data := final_payload.get("request")) and (
    #     generation_config := request_data.get("generationConfig")
    # ):
    #     if thinking_config := generation_config.get("thinkingConfig", {}):
    #         final_payload["request"]["generationConfig"]["thinkingConfig"].update(
    #             {"includeThoughts": False}
    #         )

    final_post_data = json.dumps(final_payload, ensure_ascii=False)

    if settings.DEBUG:
        import copy

        debug_payload = copy.deepcopy(final_payload)

        request_data = debug_payload.get("request", {})
        contents_list = request_data.get("contents", [])

        if isinstance(contents_list, list):
            for i, content_item in enumerate(contents_list):
                if isinstance(content_item, str):
                    # Handle the user's case of a list of strings
                    contents_list[i] = "<REDACTED>"
                elif isinstance(content_item, dict):
                    # Handle the standard case of a list of Content objects
                    parts_list = content_item.get("parts", [])
                    if isinstance(parts_list, list):
                        for part in parts_list:
                            if isinstance(part, dict):
                                if "text" in part:
                                    part["text"] = "<REDACTED>"
                                # Also redact inline data for images, etc.
                                if "inlineData" in part and isinstance(
                                    part.get("inlineData"), dict
                                ):
                                    if "data" in part.get("inlineData", {}):
                                        part["inlineData"]["data"] = "<REDACTED>"

        logging.info(
            f"DEBUG: Sending request to Google: {json.dumps(debug_payload, ensure_ascii=False)}"
        )

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            target_url, data=final_post_data, headers=request_headers
        )

        if settings.DEBUG:
            logging.info(f"[Status Code {resp.status_code}]Response: {resp.text}")

    if resp.status_code != 200:
        error_message = f"Upstream API error: {resp.status_code}"
        try:
            error_details = resp.json()
            error_message = error_details.get("error", {}).get("message", resp.text)
        except json.JSONDecodeError:
            error_message = resp.text

        logging.error(
            f"Request to {target_url} failed: {resp.status_code} - {error_message}"
        )
        raise Exception(error_message)

    if is_streaming:
        return StreamingResponse(
            _stream_generator(resp), media_type="text/event-stream"
        )
    else:
        try:
            response_text = resp.text
            if response_text.startswith("data: "):
                response_text = response_text[len("data: ") :]

            google_api_response = json.loads(response_text)
            standard_gemini_response = google_api_response.get("response")

            return Response(
                content=json.dumps(standard_gemini_response, ensure_ascii=False),
                status_code=200,
                media_type="application/json; charset=utf-8",
            )
        except json.JSONDecodeError:
            logging.error(
                f"Failed to decode JSON from Google API response: {resp.text}"
            )
            raise Exception("Failed to decode API response.")


async def _stream_generator(resp):
    async for chunk in resp.aiter_lines():
        if chunk:
            line = chunk
            if line.startswith("data: "):
                try:
                    obj = json.loads(line[len("data: ") :])
                    if "response" in obj:
                        response_chunk = json.dumps(
                            obj["response"], separators=(",", ":"), ensure_ascii=False
                        )
                        yield f"data: {response_chunk}\n\n"
                except json.JSONDecodeError:
                    continue


# def adjust_thinking_config(thinking_config: dict) -> dict:
#     thinking_budget = int(thinking_config.get("thinkingBudget", 0))
#     include_thoughts = thinking_config.get("includeThoughts", False)

#     final_thinking_budget = thinking_budget

#     if final_thinking_budget != 0:
#         include_thoughts = True

#     # If the thinkingBudget is between 0 and 128, we don't want to include thoughts.
#     if 0 <= thinking_budget <= 128:
#         include_thoughts = False

#     # If the thinkingBudget is -1, enable thoughts.
#     if thinking_budget <= -1:
#         final_thinking_budget = -1

#     logging.info(f"Setting thinking budget to {final_thinking_budget}")
#     logging.info(f"Setting include thoughts to {include_thoughts}")

#     thinking_config.update(
#         {
#             "includeThoughts": include_thoughts,
#             "thinkingBudget": final_thinking_budget,
#         }
#     )

#     return thinking_config


def build_gemini_payload_from_openai(openai_payload: dict) -> dict:
    model = openai_payload.get("model")
    safety_settings = openai_payload.get("safetySettings", DEFAULT_SAFETY_SETTINGS)
    request_data = {
        k: v
        for k, v in {
            "contents": openai_payload.get("contents"),
            "systemInstruction": openai_payload.get("systemInstruction"),
            "cachedContent": openai_payload.get("cachedContent"),
            "tools": openai_payload.get("tools"),
            "toolConfig": openai_payload.get("toolConfig"),
            "safetySettings": safety_settings,
            "generationConfig": openai_payload.get("generationConfig", {}),
        }.items()
        if v is not None
    }
    return {"model": model, "request": request_data}


def build_gemini_payload_from_native(
    native_request: dict, model_from_path: str
) -> dict:
    if "safetySettings" not in native_request:
        native_request["safetySettings"] = DEFAULT_SAFETY_SETTINGS
    return {"model": model_from_path, "request": native_request}

import asyncio
import json
import logging

import httpx
from fastapi import Response
from fastapi.responses import StreamingResponse

from .auth import get_user_project_id, onboard_user
from .credential_manager import ManagedCredential
from .constants import CODE_ASSIST_ENDPOINT, DEFAULT_SAFETY_SETTINGS
from .models import GeminiRequest, GeminiResponse
from .settings import settings
from .utils import get_user_agent, create_redacted_payload


async def send_gemini_request(
    managed_cred: ManagedCredential, payload: dict, is_streaming: bool = False
) -> Response:
    creds = managed_cred.credential
    if not creds:
        raise Exception("Invalid credential object provided.")

    # Pass the managed_cred object to the helper functions
    proj_id = await get_user_project_id(managed_cred)
    if not proj_id:
        raise Exception("Failed to get user project ID.")

    await onboard_user(managed_cred)

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

    final_post_data = json.dumps(final_payload, ensure_ascii=False)

    if settings.DEBUG:
        logging.info(f"--- Credential Details ---")
        logging.info(f"Project ID: {proj_id}")
        if managed_cred.user_email:
            logging.info(f"User Email: {managed_cred.user_email}")
        if managed_cred.credential and managed_cred.credential.refresh_token:
            token_snippet = managed_cred.credential.refresh_token[-5:]
            logging.info(f"Credential Used (Refresh Token ending in): ...{token_snippet}")
        logging.info(f"--- Upstream Request to Google ---")
        logging.info(f"URL: {target_url}")
        logging.info(f"Headers: {json.dumps(request_headers, indent=2)}")
        if settings.DEBUG_REDACT_LOGS:
            redacted_payload = create_redacted_payload(final_payload)
            logging.info(f"Payload: {json.dumps(redacted_payload, indent=2, ensure_ascii=False)}")
        else:
            logging.info(f"Payload: {json.dumps(final_payload, indent=2, ensure_ascii=False)}")
        logging.info("------------------------------------")

    async with httpx.AsyncClient(timeout=300) as client:
        resp = await client.post(
            target_url, data=final_post_data, headers=request_headers
        )

    if settings.DEBUG:
        logging.info(f"--- Upstream Response from Google ---")
        logging.info(f"Status Code: {resp.status_code}")
        logging.info(f"Headers: {json.dumps(dict(resp.headers), indent=2)}")
        try:
            # Try to pretty-print JSON if possible, otherwise print raw text
            response_json = resp.json()
            logging.info(f"Body: {json.dumps(response_json, indent=2, ensure_ascii=False)}")
        except json.JSONDecodeError:
            logging.info(f"Body: {resp.text}")
        logging.info("-------------------------------------")

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
            
            validated_response = GeminiResponse.model_validate(standard_gemini_response)

            return Response(
                content=validated_response.model_dump_json(),
                status_code=200,
                media_type="application/json; charset=utf-8",
            )
        except json.JSONDecodeError:
            logging.error(
                f"Failed to decode JSON from Google API response: {resp.text}"
            )
            raise Exception("Failed to decode API response.")
        except Exception as e:
            logging.error(f"Error validating Gemini response: {e}")
            raise Exception("Failed to validate API response.")


async def _stream_generator(resp):
    async for chunk in resp.aiter_lines():
        if chunk:
            line = chunk
            if line.startswith("data: "):
                try:
                    obj = json.loads(line[len("data: ") :])
                    if "response" in obj:
                        validated_chunk = GeminiResponse.model_validate(obj["response"])
                        response_chunk = validated_chunk.model_dump_json()
                        yield f"data: {response_chunk}\n\n"
                except (json.JSONDecodeError, Exception):
                    continue

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
    validated_request = GeminiRequest.model_validate(native_request)
    
    if "safetySettings" not in validated_request.model_dump(exclude_unset=True):
        validated_request.safetySettings = DEFAULT_SAFETY_SETTINGS
        
    return {"model": model_from_path, "request": validated_request.model_dump(exclude_unset=True)}


import json
from typing import Any, Dict, Callable

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import ValidationError

from .auth import authenticate_user
from .constants import DEFAULT_SAFETY_SETTINGS, SUPPORTED_MODELS
from .credential_manager import ManagedCredential
from .dependencies import get_validated_credential, get_request_body
from .google_api_client import prepare_credential, send_gemini_request
from .logger import get_logger
from .models import (
    CountTokensResponse,
    EmbedContentResponse,
    GeminiRequest,
    BatchEmbedContentResponse,
    GeminiResponse,
)
from .response_handler import process_upstream_response
from .streaming import format_as_gemini_sse
from .utils import build_gemini_url

logger = get_logger(__name__)
router = APIRouter()


async def handle_request(
    model_name: str,
    action: str,
    managed_cred: ManagedCredential,
    gemini_request_body: Dict[str, Any],
    is_streaming: bool,
    response_model: Any = None,
    response_formatter: Callable[..., str] = format_as_gemini_sse,
    formatter_context: Dict[str, Any] = None,
    response_transformer: Callable[[GeminiResponse, Any], Any] = None,
    original_request: Any = None,
):
    """Generic handler for all actions."""
    streaming_status = "Streaming" if is_streaming else "Non-Streaming"
    logger.info(
        f"Handling Proxy Request for model '{model_name}' with action '{action}' ({streaming_status})"
    )

    if "safetySettings" not in gemini_request_body:
        gemini_request_body["safetySettings"] = DEFAULT_SAFETY_SETTINGS

    project_id = await prepare_credential(managed_cred)

    target_url = build_gemini_url(action)

    final_payload = {
        "model": model_name,
        "project": project_id,
        "request": gemini_request_body,
    }

    upstream_response = await send_gemini_request(
        managed_cred, target_url, final_payload
    )

    if response_model and not is_streaming:
        try:
            validated_response = response_model.model_validate(upstream_response.json())
            return Response(
                content=validated_response.model_dump_json(),
                status_code=200,
                media_type="application/json; charset=utf-8",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error processing {action} response: {e}"
            )

    return await process_upstream_response(
        upstream_response=upstream_response,
        is_streaming=is_streaming,
        response_formatter=response_formatter,
        formatter_context=formatter_context,
        response_transformer=response_transformer,
        original_request=original_request,
    )


async def handle_gemini_request(
    model_name: str,
    action: str,
    managed_cred: ManagedCredential,
    incoming_request_data: dict = Depends(get_request_body),
    response_model: Any = None,
):
    """Handler for native Gemini requests."""
    is_streaming = "stream" in action.lower()
    try:
        gemini_request = GeminiRequest.model_validate(incoming_request_data)
        gemini_request_body = gemini_request.model_dump(exclude_unset=True)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=f"Request validation failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    return await handle_request(
        model_name=model_name,
        action=action,
        managed_cred=managed_cred,
        gemini_request_body=gemini_request_body,
        is_streaming=is_streaming,
        response_model=response_model,
    )


@router.get("/models")
async def list_models(_: str = Depends(authenticate_user)):
    models_response = {"models": SUPPORTED_MODELS}
    return Response(
        content=json.dumps(models_response, ensure_ascii=False),
        status_code=200,
        media_type="application/json; charset=utf-8",
    )


@router.post("/models/{model_name:path}:generateContent")
async def generate_content(
    model_name: str,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
    incoming_request_data: dict = Depends(get_request_body),
):
    return await handle_gemini_request(
        model_name, "generateContent", managed_cred, incoming_request_data
    )


@router.post("/models/{model_name:path}:streamGenerateContent")
async def stream_generate_content(
    model_name: str,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
    incoming_request_data: dict = Depends(get_request_body),
):
    return await handle_gemini_request(
        model_name, "streamGenerateContent", managed_cred, incoming_request_data
    )


@router.post("/models/{model_name:path}:countTokens")
async def count_tokens(
    model_name: str,  # This is from the URL, e.g., gemini-2.5-pro
    managed_cred: ManagedCredential = Depends(get_validated_credential),
    incoming_payload: dict = Depends(get_request_body),
):
    """
    Handles the countTokens request with its specific payload structure.
    """
    final_payload = {
        "request": {
            "model": model_name,
            **incoming_payload,
        }
    }

    await prepare_credential(managed_cred)

    target_url = build_gemini_url("countTokens")

    upstream_response = await send_gemini_request(
        managed_cred, target_url, final_payload
    )

    try:
        validated_response = CountTokensResponse.model_validate(
            upstream_response.json()
        )
        return Response(
            content=validated_response.model_dump_json(),
            status_code=200,
            media_type="application/json; charset=utf-8",
        )
    except (ValidationError, json.JSONDecodeError) as e:
        logger.error(f"Error processing countTokens response: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing countTokens response: {e}"
        )


@router.post("/models/{model_name:path}:embedContent")
async def embed_content(
    model_name: str,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
    incoming_request_data: dict = Depends(get_request_body),
):
    return await handle_gemini_request(
        model_name,
        "embedContent",
        managed_cred,
        incoming_request_data,
        response_model=EmbedContentResponse,
    )


@router.post("/models/{model_name:path}:batchEmbedContents")
async def batch_embed_contents(
    model_name: str,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
    incoming_payload: dict = Depends(get_request_body),
):
    """
    Handles the batchEmbedContents request with its specific payload structure.
    """
    await prepare_credential(managed_cred)

    target_url = build_gemini_url("batchEmbedContents")

    upstream_response = await send_gemini_request(
        managed_cred, target_url, incoming_payload
    )

    try:
        validated_response = BatchEmbedContentResponse.model_validate(
            upstream_response.json()
        )
        return Response(
            content=validated_response.model_dump_json(),
            status_code=200,
            media_type="application/json; charset=utf-8",
        )
    except (ValidationError, json.JSONDecodeError) as e:
        logger.error(f"Error processing batchEmbedContents response: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing batchEmbedContents response: {e}"
        )

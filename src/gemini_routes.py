import json
from typing import Any, Dict
from fastapi import APIRouter, Request, Response, Depends, HTTPException
from pydantic import ValidationError

from .auth import authenticate_user
from .dependencies import get_validated_credential
from .google_api_client import send_gemini_request, prepare_credential
from .constants import SUPPORTED_MODELS, DEFAULT_SAFETY_SETTINGS
from .models import GeminiRequest, CountTokensResponse, EmbedContentResponse
from .credential_manager import ManagedCredential
from .response_handler import process_upstream_response
from .streaming import format_as_gemini_sse
from .logger import get_logger
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
    response_formatter: Any = format_as_gemini_sse,
    formatter_context: Dict[str, Any] = None,
    openai_request: Any = None,
):
    """Generic handler for all actions."""
    streaming_status = "Streaming" if is_streaming else "Non-Streaming"
    logger.info(f"Handling Proxy Request for model '{model_name}' with action '{action}' ({streaming_status})")

    if 'safetySettings' not in gemini_request_body:
        gemini_request_body['safetySettings'] = DEFAULT_SAFETY_SETTINGS

    project_id = await prepare_credential(managed_cred)

    target_url = build_gemini_url(action)

    final_payload = {
        "model": model_name,
        "project": project_id,
        "request": gemini_request_body,
    }

    upstream_response = await send_gemini_request(managed_cred, target_url, final_payload)

    if response_model and not is_streaming:
        try:
            validated_response = response_model.model_validate(upstream_response.json())
            return Response(
                content=validated_response.model_dump_json(),
                status_code=200,
                media_type="application/json; charset=utf-8",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing {action} response: {e}")

    return await process_upstream_response(
        upstream_response=upstream_response,
        is_streaming=is_streaming,
        response_formatter=response_formatter,
        formatter_context=formatter_context,
        openai_request=openai_request,
    )

async def handle_gemini_request(
    request: Request,
    model_name: str,
    action: str,
    managed_cred: ManagedCredential,
    response_model: Any = None,
):
    """Handler for native Gemini requests."""
    is_streaming = "stream" in action.lower()
    try:
        post_data = await request.body()
        incoming_request_data = json.loads(post_data) if post_data else {}
        gemini_request = GeminiRequest.model_validate(incoming_request_data)
        gemini_request_body = gemini_request.model_dump(exclude_unset=True)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in request body: {e}")
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
    request: Request,
    model_name: str,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    return await handle_gemini_request(request, model_name, "generateContent", managed_cred)

@router.post("/models/{model_name:path}:streamGenerateContent")
async def stream_generate_content(
    request: Request,
    model_name: str,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    return await handle_gemini_request(request, model_name, "streamGenerateContent", managed_cred)

@router.post("/models/{model_name:path}:countTokens")
async def count_tokens(
    request: Request,
    model_name: str,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    return await handle_gemini_request(
        request, model_name, "countTokens", managed_cred, response_model=CountTokensResponse
    )

@router.post("/models/{model_name:path}:embedContent")
async def embed_content(
    request: Request,
    model_name: str,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    return await handle_gemini_request(
        request, model_name, "embedContent", managed_cred, response_model=EmbedContentResponse
    )

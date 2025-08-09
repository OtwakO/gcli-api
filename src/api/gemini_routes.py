import json
import uuid

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from ..adapters.formatters import GeminiFormatter
from ..core.auth import authenticate_user
from ..core.credential_manager import ManagedCredential
from ..core.google_api_client import prepare_credential, send_gemini_request
from ..models.gemini import CountTokensResponse, GeminiRequest
from ..services.embedding_service import embedding_service
from ..utils.constants import SUPPORTED_MODELS
from ..utils.logger import get_logger
from ..utils.utils import build_gemini_url
from .dependencies import get_request_body, get_validated_credential
from .response_handler import handle_request

logger = get_logger(__name__)
router = APIRouter()


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
    gemini_request: GeminiRequest,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    formatter = GeminiFormatter(
        {"response_id": f"chatcmpl-{uuid.uuid4()}", "model": model_name}
    )

    return await handle_request(
        model_name=model_name,
        action="generateContent",
        managed_cred=managed_cred,
        gemini_request_body=gemini_request,
        is_streaming=False,
        formatter=formatter,
    )


@router.post("/models/{model_name:path}:streamGenerateContent")
async def stream_generate_content(
    model_name: str,
    gemini_request: GeminiRequest,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    formatter = GeminiFormatter(
        {"response_id": f"chatcmpl-{uuid.uuid4()}", "model": model_name}
    )

    return await handle_request(
        model_name=model_name,
        action="streamGenerateContent",
        managed_cred=managed_cred,
        gemini_request_body=gemini_request,
        is_streaming=True,
        formatter=formatter,
    )


@router.post("/models/{model_name:path}:countTokens")
async def count_tokens(
    model_name: str,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
    incoming_payload: dict = Depends(get_request_body),
):
    """Handles the countTokens request with its specific payload structure."""
    try:
        await prepare_credential(managed_cred)
        target_url = build_gemini_url("countTokens", model_name)

        final_payload = {"request": {"model": model_name, **incoming_payload}}

        upstream_response = await send_gemini_request(
            managed_cred, target_url, final_payload
        )

        validated_response = CountTokensResponse.model_validate(
            upstream_response.json()
        )
        return JSONResponse(
            content=validated_response.model_dump(exclude_unset=True),
            status_code=200,
        )
    except (ValidationError, json.JSONDecodeError) as e:
        logger.error(f"Error processing countTokens response: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail=f"Error processing countTokens response: {e}"
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred in countTokens: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post("/models/{model_name:path}:embedContent")
async def embed_content(
    model_name: str,
    incoming_payload: dict = Depends(get_request_body),
    _user: str = Depends(authenticate_user),
):
    """Handles the native embedContent request via the EmbeddingService."""
    validated_response = await embedding_service.execute_embedding_request(
        action="embedContent",
        model_name=model_name,
        payload=incoming_payload,
    )
    return JSONResponse(
        content=validated_response.model_dump(exclude_unset=True),
        status_code=200,
    )


@router.post("/models/{model_name:path}:batchEmbedContents")
async def batch_embed_contents(
    model_name: str,
    incoming_payload: dict = Depends(get_request_body),
    _user: str = Depends(authenticate_user),
):
    """Handles the native batchEmbedContents request via the EmbeddingService."""
    validated_response = await embedding_service.execute_embedding_request(
        action="batchEmbedContents",
        model_name=model_name,
        payload=incoming_payload,
    )
    return JSONResponse(
        content=validated_response.model_dump(exclude_unset=True),
        status_code=200,
    )

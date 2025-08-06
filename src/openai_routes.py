import uuid

from fastapi import APIRouter, Depends, HTTPException

from .auth import authenticate_user
from .constants import SUPPORTED_MODELS
from .credential_manager import ManagedCredential
from .dependencies import get_validated_credential
from .gemini_routes import handle_request
from .google_api_client import prepare_credential, send_gemini_request
from .logger import get_logger
from .models import (
    EmbedContentResponse,
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
)
from .openai_transformers import (
    gemini_embedding_response_to_openai,
    openai_embedding_request_to_gemini,
    openai_request_to_gemini,
    gemini_response_to_openai,
)
from .streaming import format_as_openai_sse
from .utils import build_gemini_url

logger = get_logger(__name__)
router = APIRouter()


@router.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def openai_embeddings(
    request: OpenAIEmbeddingRequest,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    """Handles OpenAI-compatible embedding requests."""
    try:
        model_name = f"models/{request.model}"
        gemini_request_body = openai_embedding_request_to_gemini(request)

        project_id = await prepare_credential(managed_cred)
        target_url = build_gemini_url("embedContent")

        final_payload = {
            "model": model_name,
            "project": project_id,
            "request": gemini_request_body,
        }

        upstream_response = await send_gemini_request(
            managed_cred, target_url, final_payload
        )

        gemini_response_obj = upstream_response.json()

        validated_gemini_response = EmbedContentResponse.model_validate(
            gemini_response_obj
        )

        openai_response = gemini_embedding_response_to_openai(
            validated_gemini_response, request.model
        )

        return openai_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing OpenAI embedding request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


@router.post("/v1/chat/completions", response_model=OpenAIChatCompletionResponse)
async def openai_chat_completions(
    request: OpenAIChatCompletionRequest,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    try:
        gemini_request_body = openai_request_to_gemini(request)
    except Exception as e:
        logger.error(f"Error processing OpenAI chat completions request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    action = "streamGenerateContent" if request.stream else "generateContent"

    formatter_context = {
        "response_id": f"chatcmpl-{uuid.uuid4()}",
        "model": request.model,
        "is_openai": True,
    }

    return await handle_request(
        model_name=request.model,
        action=action,
        managed_cred=managed_cred,
        gemini_request_body=gemini_request_body,
        is_streaming=request.stream,
        response_formatter=format_as_openai_sse,
        formatter_context=formatter_context,
        response_transformer=gemini_response_to_openai,
        original_request=request,
    )


@router.get("/v1/models")
async def openai_list_models(_: str = Depends(authenticate_user)):
    openai_models = [
        {
            "id": model["name"].replace("models/", ""),
            "object": "model",
            "created": 1677610602,  # Static timestamp
            "owned_by": "google",
        }
        for model in SUPPORTED_MODELS
    ]
    return {"object": "list", "data": openai_models}

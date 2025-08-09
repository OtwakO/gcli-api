import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from ..adapters.adapters import openai_adapter, openai_embedding_adapter
from ..core.auth import authenticate_user
from ..core.credential_manager import ManagedCredential
from ..models.openai import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
)
from ..services.embedding_service import embedding_service
from ..utils.constants import SUPPORTED_MODELS
from ..utils.logger import get_logger
from .dependencies import get_validated_credential
from .response_handler import handle_request

logger = get_logger(__name__)
router = APIRouter()


@router.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def openai_embeddings(
    request: OpenAIEmbeddingRequest, _: str = Depends(authenticate_user)
):
    """Handles OpenAI-compatible embedding requests via the EmbeddingService."""
    try:
        # 1. Adapt the incoming request to the internal format
        action, model_name, gemini_request_body = (
            openai_embedding_adapter.request_transformer(request)
        )

        # 2. Delegate the core logic to the EmbeddingService
        validated_gemini_response = await embedding_service.execute_embedding_request(
            action=action,
            model_name=model_name,
            payload=gemini_request_body,
        )

        # 3. Format the service's response back to the OpenAI format
        formatter = openai_embedding_adapter.formatter_class({})
        return formatter.format_response(validated_gemini_response, request)

    except (HTTPException, ValidationError):
        # Re-raise exceptions that are already handled or structured
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
        model_name, gemini_request = openai_adapter.request_transformer(request)
    except Exception as e:
        logger.error(
            f"Error processing OpenAI chat completions request: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

    action = "streamGenerateContent" if request.stream else "generateContent"

    formatter_context = {
        "response_id": f"chatcmpl-{uuid.uuid4()}",
        "model": request.model,
    }
    formatter = openai_adapter.formatter_class(formatter_context)

    return await handle_request(
        model_name=model_name,
        action=action,
        managed_cred=managed_cred,
        gemini_request_body=gemini_request,
        is_streaming=request.stream,
        formatter=formatter,
        original_request=request,
    )


@router.get("/v1/models")
async def openai_list_models(_: str = Depends(authenticate_user)):
    openai_models = [
        {
            "id": model["name"],
            "object": "model",
            "created": 1677610602,  # Static timestamp
            "owned_by": "google",
        }
        for model in SUPPORTED_MODELS
    ]
    return {"object": "list", "data": openai_models}

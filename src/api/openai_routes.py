from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from ..adapters.adapters import openai_adapter, openai_embedding_adapter
from ..adapters.formatters import FormatterContext
from ..core.credential_manager import ManagedCredential
from ..core.proxy_auth import authenticate_user
from ..models.openai import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
)
from ..services.chat_completion_service import chat_completion_service
from ..services.embedding_service import embedding_service
from ..utils.constants import SUPPORTED_MODELS
from ..utils.logger import get_logger
from ..utils.utils import generate_response_id
from .dependencies import get_validated_credential

logger = get_logger(__name__)
router = APIRouter()


@router.post("/v1/embeddings", response_model=OpenAIEmbeddingResponse)
async def openai_embeddings(
    request: OpenAIEmbeddingRequest, _: str = Depends(authenticate_user)
):
    """Handles OpenAI-compatible embedding requests via the EmbeddingService."""
    try:
        action, model_name, gemini_request_body = (
            openai_embedding_adapter.request_transformer(request)
        )
        validated_gemini_response = await embedding_service.execute_embedding_request(
            action=action,
            model_name=model_name,
            payload=gemini_request_body,
        )
        # The embedding formatter does not require specific context, but we pass
        # a compliant object for consistency.
        formatter_context = FormatterContext(response_id="", model=model_name)
        formatter = openai_embedding_adapter.formatter_class(formatter_context)
        return formatter.format_response(validated_gemini_response, request)

    except (HTTPException, ValidationError):
        raise
    except Exception as e:
        logger.error(f"Error processing OpenAI embedding request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An unexpected internal server error occurred.",
        )


@router.post("/v1/chat/completions", response_model=OpenAIChatCompletionResponse)
async def openai_chat_completions(
    request: OpenAIChatCompletionRequest,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    try:
        model_name, gemini_request = openai_adapter.request_transformer(request)

        formatter_context = FormatterContext(
            response_id=generate_response_id("chatcmpl"),
            model=request.model,
        )
        formatter = openai_adapter.formatter_class(formatter_context)

        return await chat_completion_service.handle_chat_request(
            model_name=model_name,
            managed_cred=managed_cred,
            gemini_request_body=gemini_request,
            is_streaming=request.stream,
            formatter=formatter,
            source_api="OpenAI-compatible",
            original_request=request,
        )
    except Exception as e:
        logger.error(
            f"Error processing OpenAI chat completions request: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500,
            detail="An unexpected internal server error occurred.",
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

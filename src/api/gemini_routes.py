import json

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import JSONResponse

from ..adapters.formatters import FormatterContext, GeminiFormatter
from ..core.credential_manager import ManagedCredential
from ..core.exceptions import MalformedContentError, UpstreamHttpError
from ..core.proxy_auth import authenticate_user
from ..models.gemini import (
    BatchEmbedContentsRequest,
    CountTokensRequest,
    EmbedContentRequest,
    GeminiRequest,
)
from ..services.chat_completion_service import chat_completion_service
from ..services.embedding_service import embedding_service
from ..services.model_service import model_service
from ..utils.constants import SUPPORTED_MODELS
from ..utils.logger import get_logger
from ..utils.utils import generate_response_id, sanitize_gemini_tools
from .dependencies import get_validated_credential

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
    try:
        formatter_context = FormatterContext(
            response_id=generate_response_id("chatcmpl"), model=model_name
        )
        formatter = GeminiFormatter(formatter_context)

        gemini_request.tools = sanitize_gemini_tools(gemini_request.tools)

        return await chat_completion_service.handle_chat_request(
            model_name=model_name,
            managed_cred=managed_cred,
            gemini_request_body=gemini_request,
            is_streaming=False,
            formatter=formatter,
            source_api="Native Gemini",
        )
    except (UpstreamHttpError, MalformedContentError):
        raise
    except Exception as e:
        logger.error(
            f"Error Processing Gemini generate content request: {e}", exc_info=True
        )
        raise HTTPException(
            status_code=500, detail="An unexpected internal server error occurred."
        )


@router.post("/models/{model_name:path}:streamGenerateContent")
async def stream_generate_content(
    model_name: str,
    gemini_request: GeminiRequest,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    try:
        formatter_context = FormatterContext(
            response_id=generate_response_id("chatcmpl"), model=model_name
        )
        formatter = GeminiFormatter(formatter_context)

        gemini_request.tools = sanitize_gemini_tools(gemini_request.tools)

        return await chat_completion_service.handle_chat_request(
            model_name=model_name,
            managed_cred=managed_cred,
            gemini_request_body=gemini_request,
            is_streaming=True,
            formatter=formatter,
            source_api="Native Gemini",
        )
    except (UpstreamHttpError, MalformedContentError):
        raise
    except Exception as e:
        logger.error(
            f"Error Processing Gemini stream generate content request: {e}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="An unexpected internal server error occurred."
        )


@router.post("/models/{model_name:path}:countTokens")
async def count_tokens(
    model_name: str,
    request_body: CountTokensRequest,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    """Handles the countTokens request by delegating to the ModelService."""
    payload = request_body.model_dump(exclude_unset=True)
    validated_response = await model_service.count_tokens(
        model_name, managed_cred, payload
    )
    return JSONResponse(
        content=validated_response.model_dump(exclude_unset=True),
        status_code=200,
    )


@router.post("/models/{model_name:path}:embedContent")
async def embed_content(
    model_name: str,
    request_body: EmbedContentRequest,
    _user: str = Depends(authenticate_user),
):
    """Handles the native embedContent request via the EmbeddingService."""
    payload = request_body.model_dump(exclude_unset=True)
    validated_response = await embedding_service.execute_embedding_request(
        action="embedContent",
        model_name=model_name,
        payload=payload,
    )
    return JSONResponse(
        content=validated_response.model_dump(exclude_unset=True),
        status_code=200,
    )


@router.post("/models/{model_name:path}:batchEmbedContents")
async def batch_embed_contents(
    model_name: str,
    request_body: BatchEmbedContentsRequest,
    _user: str = Depends(authenticate_user),
):
    """Handles the native batchEmbedContents request via the EmbeddingService."""
    payload = request_body.model_dump(exclude_unset=True)
    validated_response = await embedding_service.execute_embedding_request(
        action="batchEmbedContents",
        model_name=model_name,
        payload=payload,
    )
    return JSONResponse(
        content=validated_response.model_dump(exclude_unset=True),
        status_code=200,
    )

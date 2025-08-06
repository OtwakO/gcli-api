import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from .auth import authenticate_user
from .dependencies import get_validated_credential
from .constants import SUPPORTED_MODELS
from .credential_manager import ManagedCredential
from .google_api_client import send_gemini_request
from .models import OpenAIChatCompletionRequest, OpenAIChatCompletionResponse, GeminiResponse
from .openai_transformers import openai_request_to_gemini, gemini_response_to_openai
from .settings import settings
from .streaming import (
    process_stream_for_client,
    format_as_openai_sse,
    wrap_thoughts_in_gemini_response,
)
from .logger import get_logger, format_log

logger = get_logger(__name__)
router = APIRouter()


from .response_handler import process_upstream_response

@router.post("/v1/chat/completions", response_model=OpenAIChatCompletionResponse)
async def openai_chat_completions(
    request: OpenAIChatCompletionRequest,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    streaming_status = "Streaming" if request.stream else "Non-Streaming"
    logger.info(f"Handling OpenAI Chat Completions Request ({streaming_status})")
    
    try:
        # This builds the main body of the request.
        gemini_request_body = openai_request_to_gemini(request)
    except Exception as e:
        logger.error(f"Error processing OpenAI request: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Request processing failed: {e}")

    # The send_gemini_request function expects a payload with top-level 'model' and 'request' keys.
    final_payload = {
        "model": request.model,
        "request": gemini_request_body,
    }

    upstream_response = await send_gemini_request(
        managed_cred, final_payload, is_streaming=request.stream
    )

    formatter_context = {
        "response_id": f"chatcmpl-{uuid.uuid4()}",
        "model": request.model,
        "is_openai": True,
    }

    return await process_upstream_response(
        upstream_response=upstream_response,
        is_streaming=request.stream,
        response_formatter=format_as_openai_sse,
        formatter_context=formatter_context,
        openai_request=request,
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
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

    if request.stream:
        formatter_context = {
            "response_id": f"chatcmpl-{uuid.uuid4()}",
            "model": request.model,
            "is_openai": True,
        }
        stream_processor = process_stream_for_client(
            upstream_response, 
            format_as_openai_sse,
            formatter_context,
        )
        return StreamingResponse(stream_processor, media_type="text/event-stream")
    else:
        gemini_response = GeminiResponse.model_validate(upstream_response.json()["response"])

        if settings.THOUGHT_WRAPPER_TAGS and len(settings.THOUGHT_WRAPPER_TAGS) == 2:
            gemini_response = wrap_thoughts_in_gemini_response(
                gemini_response, settings.THOUGHT_WRAPPER_TAGS
            )

        openai_response_dict = gemini_response_to_openai(gemini_response, request.model)
        openai_response = OpenAIChatCompletionResponse.model_validate(openai_response_dict)

        if settings.DEBUG:
            logger.debug(format_log(
                "Sending to Client (Non-Streaming)", 
                openai_response.model_dump(exclude_unset=True),
                is_json=True
            ))

        return openai_response


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
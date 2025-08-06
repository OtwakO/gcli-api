import uuid

from fastapi import APIRouter, Depends, HTTPException

from .auth import authenticate_user
from .constants import SUPPORTED_MODELS
from .credential_manager import ManagedCredential
from .dependencies import get_validated_credential
from .google_api_client import prepare_credential
from .gemini_routes import handle_request
from .logger import get_logger
from .models import (
    OpenAIChatCompletionRequest,
    OpenAIChatCompletionResponse,
)
from .openai_transformers import openai_request_to_gemini
from .streaming import format_as_openai_sse

logger = get_logger(__name__)
router = APIRouter()


@router.post("/v1/chat/completions", response_model=OpenAIChatCompletionResponse)
async def openai_chat_completions(
    request: OpenAIChatCompletionRequest,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    try:
        gemini_request_body = openai_request_to_gemini(request)
    except Exception as e:
        logger.error(f"Error processing OpenAI request: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Request processing failed: {e}")

    project_id = await prepare_credential(managed_cred)

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

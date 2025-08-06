import json
import re
from fastapi import APIRouter, Request, Response, Depends, HTTPException
from fastapi.responses import StreamingResponse

from .auth import authenticate_user
from .dependencies import get_validated_credential
from .google_api_client import send_gemini_request
from .constants import SUPPORTED_MODELS
from .models import GeminiRequest, GeminiResponse
from .credential_manager import ManagedCredential
from .settings import settings
from .streaming import (
    process_stream_for_client,
    format_as_gemini_sse,
    wrap_thoughts_in_gemini_response,
)
from .logger import get_logger, format_log

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

from .response_handler import process_upstream_response

@router.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def gemini_proxy(
    request: Request,
    full_path: str,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    post_data = await request.body()
    is_streaming = "stream" in full_path.lower()
    streaming_status = "Streaming" if is_streaming else "Non-Streaming"
    logger.info(f"Handling Gemini Proxy Request ({streaming_status})")

    # This regex needs to handle potential query parameters in the path.
    model_match = re.match(r"models/([^:]+):(\w+)", full_path)
    if not model_match:
        logger.error(f"Could not parse model name and action from path: '{full_path}'. The regex did not match.")
        raise HTTPException(status_code=400, detail=f"Could not extract model name from path: {full_path}")

    model_name = model_match.group(1)

    try:
        incoming_request_data = json.loads(post_data) if post_data else {}
        incoming_request = GeminiRequest.model_validate(incoming_request_data)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in request body: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request body: {e}")

    gemini_request_body = incoming_request.model_dump(exclude_unset=True)

    # Ensure default safety settings are applied if the user doesn't provide any.
    if 'safetySettings' not in gemini_request_body:
        from .constants import DEFAULT_SAFETY_SETTINGS
        gemini_request_body['safetySettings'] = DEFAULT_SAFETY_SETTINGS

    gemini_payload = {
        "model": model_name,
        "request": gemini_request_body
    }

    upstream_response = await send_gemini_request(managed_cred, gemini_payload, is_streaming=is_streaming)

    return await process_upstream_response(
        upstream_response=upstream_response,
        is_streaming=is_streaming,
        response_formatter=format_as_gemini_sse,
    )

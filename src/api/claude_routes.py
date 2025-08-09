import uuid

from fastapi import APIRouter, Depends, HTTPException

from ..adapters.adapters import claude_adapter
from ..core.credential_manager import ManagedCredential
from .dependencies import get_validated_credential
from .response_handler import handle_request
from ..utils.logger import get_logger
from ..models.claude import ClaudeMessagesRequest

logger = get_logger(__name__)
router = APIRouter()


@router.post("/v1/messages", tags=["Claude"])
async def claude_messages(
    claude_request: ClaudeMessagesRequest,
    managed_cred: ManagedCredential = Depends(get_validated_credential),
):
    """
    Handles Claude-compatible /v1/messages requests by transforming them
    for the Gemini API and then formatting the response back to the
    Claude SSE format.
    """
    try:
        is_streaming = claude_request.stream
        model_name, gemini_request_body = claude_adapter.request_transformer(
            claude_request
        )
        action = "streamGenerateContent" if is_streaming else "generateContent"

        formatter_context = {
            "response_id": f"msg_{uuid.uuid4().hex}",
            "model": model_name,
        }
        formatter = claude_adapter.formatter_class(formatter_context)

        return await handle_request(
            model_name=model_name,
            action=action,
            managed_cred=managed_cred,
            gemini_request_body=gemini_request_body,
            is_streaming=is_streaming,
            formatter=formatter,
            original_request=claude_request,
        )

    except Exception as e:
        logger.error(f"Error processing Claude request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

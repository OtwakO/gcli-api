from fastapi import APIRouter, Depends, HTTPException

from ..adapters.adapters import claude_adapter
from ..adapters.formatters import FormatterContext
from ..core.credential_manager import ManagedCredential
from ..models.claude import ClaudeMessagesRequest
from ..services.chat_completion_service import chat_completion_service
from ..utils.logger import get_logger
from ..utils.utils import generate_response_id
from .dependencies import get_validated_credential

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
        model_name, gemini_request_body = claude_adapter.request_transformer(
            claude_request
        )

        formatter_context = FormatterContext(
            response_id=generate_response_id("msg"),
            model=model_name,
        )
        formatter = claude_adapter.formatter_class(formatter_context)

        return await chat_completion_service.handle_chat_request(
            model_name=model_name,
            managed_cred=managed_cred,
            gemini_request_body=gemini_request_body,
            is_streaming=claude_request.stream,
            formatter=formatter,
            source_api="Claude-compatible",
            original_request=claude_request,
        )

    except Exception as e:
        logger.error(f"Error processing Claude request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An unexpected internal server error occurred."
        )

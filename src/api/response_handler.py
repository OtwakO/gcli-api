from typing import Any, Dict, Union

from fastapi.responses import JSONResponse, StreamingResponse

from ..adapters.formatters import Formatter
from ..core.credential_manager import ManagedCredential
from ..core.google_api_client import (
    GoogleStreamer,
    prepare_credential,
    send_gemini_request,
)
from ..core.settings import settings
from ..core.streaming import StreamProcessor
from ..models.gemini import GeminiRequest, GeminiResponse
from ..utils.constants import DEFAULT_SAFETY_SETTINGS
from ..utils.logger import format_log, get_logger
from ..utils.utils import build_gemini_url

logger = get_logger(__name__)


async def handle_request(
    model_name: str,
    action: str,
    managed_cred: ManagedCredential,
    gemini_request_body: Union[Dict[str, Any], GeminiRequest],
    is_streaming: bool,
    formatter: Formatter,
    original_request: Any = None,
):
    """Generic handler for all actions."""
    streaming_status = "Streaming" if is_streaming else "Non-Streaming"
    logger.info(
        f"Handling Proxy Request for model '{model_name}' with action '{action}' ({streaming_status})"
    )

    request_payload = (
        gemini_request_body.model_dump(exclude_unset=True)
        if isinstance(gemini_request_body, GeminiRequest)
        else gemini_request_body
    )

    if "safetySettings" not in request_payload:
        request_payload["safetySettings"] = DEFAULT_SAFETY_SETTINGS

    project_id = await prepare_credential(managed_cred)

    target_url = build_gemini_url(action, model_name)

    final_payload = {
        "model": model_name,
        "project": project_id,
        "request": request_payload,
    }

    if is_streaming:
        streamer = GoogleStreamer(managed_cred, target_url, final_payload)
        processor = StreamProcessor(streamer, formatter)
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        return StreamingResponse(
            processor.process(), media_type="text/event-stream", headers=headers
        )
    else:
        upstream_response = await send_gemini_request(
            managed_cred, target_url, final_payload
        )
        return await process_non_streaming_response(
            upstream_response=upstream_response,
            formatter=formatter,
            original_request=original_request,
        )


async def process_non_streaming_response(
    upstream_response,
    formatter: Formatter,
    original_request: Any = None,
):
    """Processes a non-streaming upstream response and returns a JSONResponse."""
    response_data = upstream_response.json()
    gemini_response_data = response_data.get("response") or response_data.get("result")
    if not gemini_response_data:
        raise ValueError("Could not find 'response' or 'result' in upstream JSON")

    gemini_response = GeminiResponse.model_validate(gemini_response_data)

    final_response_model = formatter.format_response(gemini_response, original_request)

    response_content = final_response_model.model_dump(exclude_unset=True)

    if settings.DEBUG:
        logger.debug(
            format_log(
                "Sending to Client (Non-Streaming)",
                response_content,
                is_json=True,
            )
        )

    return JSONResponse(content=response_content)

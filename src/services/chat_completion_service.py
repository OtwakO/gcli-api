from typing import Any, Dict, Union

from fastapi.responses import JSONResponse, StreamingResponse

from ..adapters.formatters import Formatter
from ..core.credential_manager import ManagedCredential
from ..core.google_api_client import send_request
from ..core.settings import settings
from ..core.streaming import StreamProcessor
from ..core.upstream_auth import OAuthStrategy
from ..models.gemini import GeminiRequest, GeminiResponse
from ..services.onboarding_service import onboarding_service
from ..utils.constants import DEFAULT_SAFETY_SETTINGS
from ..utils.logger import format_log, get_logger
from ..utils.utils import build_gemini_url, create_redacted_payload

logger = get_logger(__name__)


class ChatCompletionService:
    """
    A service dedicated to handling all chat completion requests.
    It acts as a single point of entry for the API routes and orchestrates
    the request handling process.
    """

    async def handle_chat_request(
        self,
        model_name: str,
        managed_cred: ManagedCredential,
        gemini_request_body: Union[Dict[str, Any], GeminiRequest],
        is_streaming: bool,
        formatter: Formatter,
        source_api: str,
        original_request: Any = None,
    ):
        """Main method to process the request."""
        streaming_status = "Streaming" if is_streaming else "Non-Streaming"
        action = "streamGenerateContent" if is_streaming else "generateContent"

        logger.info(
            f"Handling {source_api} Request for model '{model_name}' with action '{action}' ({streaming_status})"
        )

        request_payload = (
            gemini_request_body.model_dump(exclude_unset=True)
            if isinstance(gemini_request_body, GeminiRequest)
            else gemini_request_body
        )

        if settings.DEBUG:
            log_payload = (
                create_redacted_payload(request_payload)
                if settings.DEBUG_REDACT_LOGS
                else request_payload
            )
            logger.debug(
                format_log(
                    "Transformed Gemini Request Body",
                    log_payload,
                    is_json=True,
                )
            )

        if "safetySettings" not in request_payload:
            request_payload["safetySettings"] = DEFAULT_SAFETY_SETTINGS

        project_id = await onboarding_service.prepare_credential(managed_cred)
        target_url = build_gemini_url(action, model_name)

        final_payload = {
            "model": model_name,
            "project": project_id,
            "request": request_payload,
        }

        if is_streaming:
            return self._process_streaming_request(
                managed_cred, target_url, final_payload, formatter
            )
        else:
            return await self._process_non_streaming_request(
                managed_cred, target_url, final_payload, formatter, original_request
            )

    def _process_streaming_request(
        self,
        managed_cred: ManagedCredential,
        target_url: str,
        payload: Dict[str, Any],
        formatter: Formatter,
    ):
        processor = StreamProcessor(managed_cred, target_url, payload, formatter)
        headers = {
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        return StreamingResponse(
            processor.process(), media_type="text/event-stream", headers=headers
        )

    async def _process_non_streaming_request(
        self,
        managed_cred: ManagedCredential,
        target_url: str,
        payload: Dict[str, Any],
        formatter: Formatter,
        original_request: Any,
    ):
        auth_strategy = OAuthStrategy(managed_cred)
        upstream_response = await send_request(target_url, payload, auth_strategy)
        response_data = upstream_response.json()

        gemini_response_data = response_data.get("response") or response_data.get(
            "result"
        )
        if not gemini_response_data:
            raise ValueError("Could not find 'response' or 'result' in upstream JSON")

        gemini_response = GeminiResponse.model_validate(gemini_response_data)
        final_response_model = formatter.format_response(
            gemini_response, original_request
        )
        response_content = final_response_model.model_dump(exclude_unset=True)

        logger.debug(
            format_log(
                "Sending to Client (Non-Streaming)",
                response_content,
                is_json=True,
            )
        )

        return JSONResponse(content=response_content)


# Create a singleton instance of the service
chat_completion_service = ChatCompletionService()

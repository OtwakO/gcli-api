
import json
from fastapi import Response
from fastapi.responses import StreamingResponse
from .models import GeminiResponse
from .settings import settings
from .streaming import process_stream_for_client, wrap_thoughts_in_gemini_response
from .logger import get_logger, format_log

logger = get_logger(__name__)

async def process_upstream_response(
    upstream_response,
    is_streaming: bool,
    response_formatter,
    formatter_context: dict = None,
    openai_request = None,
):
    if is_streaming:
        stream_processor = process_stream_for_client(
            upstream_response, response_formatter, formatter_context or {}
        )
        return StreamingResponse(stream_processor, media_type="text/event-stream")
    else:
        gemini_response = GeminiResponse.model_validate(upstream_response.json()["response"])

        if settings.THOUGHT_WRAPPER_TAGS and len(settings.THOUGHT_WRAPPER_TAGS) == 2:
            gemini_response = wrap_thoughts_in_gemini_response(
                gemini_response, settings.THOUGHT_WRAPPER_TAGS
            )

        if openai_request:
            from .openai_transformers import gemini_response_to_openai
            from .models import OpenAIChatCompletionResponse

            openai_response_dict = gemini_response_to_openai(gemini_response, openai_request.model)
            openai_response = OpenAIChatCompletionResponse.model_validate(openai_response_dict)

            if settings.DEBUG:
                logger.debug(
                    format_log(
                        "Sending to Client (Non-Streaming)",
                        openai_response.model_dump(exclude_unset=True),
                        is_json=True,
                    )
                )
            return openai_response
        else:
            response_data = gemini_response.model_dump(exclude_unset=True)
            if settings.DEBUG:
                logger.debug(
                    format_log(
                        "Sending to Client (Non-Streaming)",
                        response_data,
                        is_json=True,
                    )
                )
            return Response(
                content=json.dumps(response_data, ensure_ascii=False),
                status_code=200,
                media_type="application/json; charset=utf-8",
            )

import json
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import StreamingResponse

from .auth import authenticate_user
from .constants import SUPPORTED_MODELS
from .google_api_client import build_gemini_payload_from_openai, send_gemini_request
from .models import OpenAIChatCompletionRequest, OpenAIChatCompletionResponse
from .openai_transformers import (
    gemini_response_to_openai,
    gemini_stream_chunk_to_openai,
    openai_request_to_gemini,
)
from .settings import settings

router = APIRouter()


@router.post("/v1/chat/completions", response_model=OpenAIChatCompletionResponse)
async def openai_chat_completions(
    request: OpenAIChatCompletionRequest,
    http_request: Request,
    username: str = Depends(authenticate_user),
):
    try:
        gemini_request_data = openai_request_to_gemini(request)
        gemini_payload = build_gemini_payload_from_openai(gemini_request_data)
    except Exception as e:
        logging.error(f"Error processing OpenAI request: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Request processing failed: {e}")

    if request.stream:
        return StreamingResponse(
            _openai_stream_generator(gemini_payload, request.model),
            media_type="text/event-stream",
        )
    else:
        try:
            if settings.DEBUG:
                logging.info("Sending non-streaming request to Gemini...")
            
            response = await send_gemini_request(gemini_payload, is_streaming=False)
            gemini_response = json.loads(response.body)

            if settings.DEBUG:
                logging.info(f"Received Gemini response: {gemini_response}")

            openai_response = gemini_response_to_openai(gemini_response, request.model)

            if settings.DEBUG:
                logging.info(f"Transformed to OpenAI response: {openai_response}")

            return openai_response
        except Exception as e:
            logging.error(f"Non-streaming request failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))


async def _openai_stream_generator(gemini_payload, model):
    response_id = "chatcmpl-" + str(uuid.uuid4())
    try:
        response = await send_gemini_request(gemini_payload, is_streaming=True)
        async for chunk in response.body_iterator:
            if isinstance(chunk, bytes):
                chunk = chunk.decode("utf-8")
            if chunk.startswith("data: "):
                try:
                    gemini_chunk = json.loads(chunk[6:])
                    if "error" in gemini_chunk:
                        yield f"data: {json.dumps(gemini_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        return
                    openai_chunk = gemini_stream_chunk_to_openai(
                        gemini_chunk, model, response_id
                    )
                    yield f"data: {json.dumps(openai_chunk)}\n\n"
                except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                    logging.warning(f"Failed to parse streaming chunk: {e}")
                    continue
        yield "data: [DONE]\n\n"
    except Exception as e:
        logging.error(f"Streaming error: {e}")
        error_data = {
            "error": {
                "message": f"Streaming failed: {e}",
                "type": "api_error",
                "code": 500,
            }
        }
        yield f"data: {json.dumps(error_data)}\n\n"
        yield "data: [DONE]\n\n"


@router.get("/v1/models")
async def openai_list_models(username: str = Depends(authenticate_user)):
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

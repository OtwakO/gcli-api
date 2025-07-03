import json
import logging
import re
import traceback
from fastapi import APIRouter, Request, Response, Depends, HTTPException

from .auth import authenticate_user
from .google_api_client import send_gemini_request, build_gemini_payload_from_native
from .constants import SUPPORTED_MODELS

router = APIRouter()

@router.get("/models")
async def list_models(request: Request, username: str = Depends(authenticate_user)):
    models_response = {"models": SUPPORTED_MODELS}
    return Response(
        content=json.dumps(models_response, ensure_ascii=False),
        status_code=200,
        media_type="application/json; charset=utf-8",
    )

@router.api_route("/{full_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def gemini_proxy(request: Request, full_path: str, username: str = Depends(authenticate_user)):
    post_data = await request.body()
    is_streaming = "stream" in full_path.lower()

    model_match = re.match(r"models/([^:]+):(\w+)", full_path)
    if not model_match:
        raise HTTPException(status_code=400, detail="Could not extract model name from path")

    model_name = model_match.group(1)

    try:
        incoming_request = json.loads(post_data) if post_data else {}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in request body: {e}")

    gemini_payload = build_gemini_payload_from_native(incoming_request, model_name)

    try:
        return await send_gemini_request(gemini_payload, is_streaming=is_streaming)
    except Exception as e:
        logging.error(f"Gemini proxy error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

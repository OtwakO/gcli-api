import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .credential_manager import credential_manager
from .gemini_routes import router as gemini_router
from .logger import format_log, get_logger, setup_logging
from .openai_routes import router as openai_router
from .settings import settings
from .ui import create_page

logger = get_logger(__name__)


class PrettyJSONResponse(Response):
    media_type = "application/json"

    def render(self, content: any) -> bytes:
        import json

        return json.dumps(
            content,
            ensure_ascii=False,
            allow_nan=False,
            indent=4,
            separators=(", ", ": "),
        ).encode("utf-8")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if credential_manager._credentials:
        logger.info(
            f"Proxy is running with {len(credential_manager._credentials)} credential(s)."
        )
    else:
        logger.warning(
            "Proxy is running without any credentials. "
            "Please run the `generate_credentials.py` script to create credentials, "
            "or configure the CREDENTIALS_JSON_LIST environment variable."
        )
    if settings.GEMINI_AUTH_PASSWORD == "123456":
        if not settings.DEBUG:
            logger.critical(
                'CRITICAL SECURITY RISK: The default authentication password is being used in a non-DEBUG environment. Please set a strong GEMINI_AUTH_PASSWORD immediately.'
            )
        else:
            logger.warning(
                'Security risk: The default authentication password is being used. Please set a strong GEMINI_AUTH_PASSWORD in your environment.'
            )
    yield


app = FastAPI(lifespan=lifespan, default_response_class=PrettyJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def debug_logging_middleware(request: Request, call_next):
    if settings.DEBUG:
        log_data = {
            "method": request.method,
            "url": str(request.url),
            "headers": dict(request.headers),
        }

        logger.debug(format_log("Incoming Request", log_data, is_json=True))

    response = await call_next(request)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handles known HTTP exceptions and returns a structured error response."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": exc.detail, "type": "api_error"}},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Handles any other unexpected exceptions to prevent crashes."""
    logging.error(f"Unhandled exception for request {request.method} {request.url}:")
    logging.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": "An unexpected internal server error occurred.",
                "type": "unexpected_error",
                "detail": str(exc),
            }
        },
    )


@app.get("/")
async def root(request: Request):
    num_credentials = len(credential_manager._credentials)
    # Correctly determine the base URL, especially behind a proxy
    api_base_url = str(request.base_url)

    if num_credentials > 0:
        status_class = "active"
        status_text = "Active"
        message = f"The proxy is running and is rotating through <strong>{num_credentials}</strong> credential(s)."
    else:
        status_class = "inactive"
        status_text = "Inactive / Action Required"
        message = (
            "The proxy is running but has no credentials loaded. "
            "Please run the <code>generate_credentials.py</code> script to authenticate."
        )

    content = f"""
        <h1>Gemini Rotating Proxy</h1>
        <div class="status {status_class}">{status_text}</div>
        <p>{message}</p>
        <div class="info-box">
            <p><strong>API Base URL</strong></p>
            <div class="code-block">
                <code id="api-base-url">{api_base_url}</code>
                <button class="copy-btn" onclick="copyToClipboard('api-base-url', this)">Copy</button>
            </div>
        </div>
        <p class="footer">This UI provides a status overview. See the documentation for API usage details.</p>
    """
    return create_page("Gemini Proxy Status", content)


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.include_router(openai_router)
app.include_router(gemini_router, prefix="/v1beta")

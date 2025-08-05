import json
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .credential_manager import credential_manager
from .gemini_routes import router as gemini_router
from .openai_routes import router as openai_router
from .settings import settings
from .ui import create_page

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logging.getLogger("httpx").setLevel(logging.WARNING)


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
        logging.info(
            f"Proxy is running with {len(credential_manager._credentials)} credential(s)."
        )
    else:
        logging.warning(
            "Proxy is running without any credentials. "
            "Please run the `generate_credentials.py` script to create credentials, "
            "or configure the CREDENTIALS_JSON_LIST environment variable."
        )
    yield


app = FastAPI(lifespan=lifespan, default_response_class=PrettyJSONResponse)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def debug_logging_middleware(request: Request, call_next):
    if settings.DEBUG:
        logging.info("--- Incoming Request ---")
        logging.info(f"Method: {request.method}")
        logging.info(f"URL: {request.url}")
        headers = json.dumps(dict(request.headers), indent=2)
        logging.info(f"Headers: {headers}")
        logging.info("------------------------")

    response = await call_next(request)
    return response


@app.exception_handler(Exception)
async def validation_exception_handler(request, err):
    logging.error(f"Unhandled exception for request {request.method} {request.url}:")
    logging.error(traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": str(err),
                "type": "unexpected_error",
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

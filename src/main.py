import logging
import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .auth import (
    get_credentials,
    get_user_project_id,
    login,
    oauth2callback,
    onboard_user,
)
from .gemini_routes import router as gemini_router
from .openai_routes import router as openai_router
from .settings import settings

logging.basicConfig(level=logging.INFO)
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
    if settings.OAUTH_CREDS_JSON:
        logging.info("OAUTH_CREDS_JSON found. Writing to credential file.")
        try:
            with open(settings.CREDENTIAL_FILE, "w") as f:
                f.write(settings.OAUTH_CREDS_JSON)
        except PermissionError:
            logging.info(
                "Skipping writing OAUTH_CREDS_JSON to file due to permission error."
            )
        except Exception as e:
            logging.error(f"Failed to write OAUTH_CREDS_JSON: {e}.")

    creds = await get_credentials()
    if creds:
        try:
            proj_id = await get_user_project_id(creds)
            if proj_id:
                await onboard_user(creds, proj_id)
            logging.info("Proxy is authenticated and ready.")
        except Exception as e:
            logging.error(
                f"Failed during startup onboarding: {e}. Re-authentication may be needed."
            )
    else:
        logging.warning(
            f"Proxy is not authenticated. Please visit {settings.DOMAIN_NAME}/login to authenticate."
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
    if await get_credentials():
        api_base_url = str(request.base_url).replace("http://", "https://")
        return Response(
            content=f"<h1>Gemini Proxy is Authenticated and Running</h1><h2>URL: {api_base_url}</h2>",
            media_type="text/html",
        )
    else:
        return Response(
            content=f'<h1>Welcome to the Gemini Proxy</h1><p>Not authenticated. <a href="/login">Click here to log in</a>.</p>',
            media_type="text/html",
        )


@app.get("/health")
def health_check():
    return {"status": "ok"}


app.add_api_route("/login", login, methods=["GET"])
app.add_api_route("/oauth2callback", oauth2callback, methods=["GET"])

app.include_router(openai_router)
app.include_router(gemini_router, prefix="/v1beta")

import asyncio
import json
import os
import re

import httpx
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from google_auth_oauthlib.flow import Flow

from ..core.settings import settings
from ..utils.constants import SCOPES
from ..utils.logger import format_log, get_logger, setup_logging
from ..utils.ui import create_page
from ..utils.utils import get_client_metadata, get_user_agent

# --- Logging Configuration ---
setup_logging()
logger = get_logger(__name__)


# --- Configuration ---
REDIRECT_URI = f"{settings.DOMAIN_NAME}/oauth2callback"
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

app = FastAPI()
auth_flow_state = {"flow": None}


def sanitize_for_filename(text: str) -> str:
    """Creates a safe filename component from a string."""
    if not text:
        return "unspecified"
    sanitized = text.replace("@", "_").replace(".", "_")
    return re.sub(r"[^a-zA-Z0-9_.-]", "", sanitized)


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serves the main page with the login form."""
    content = """
        <h1>Gemini Credential Generator</h1>
        <p>
            To generate a new credential, you can optionally specify a Google Cloud Project ID below.
            If you leave it blank, the tool will attempt to discover it automatically.
        </p>
        <form action="/login" method="post">
            <label for="project_id">Google Cloud Project ID (Optional)</label>
            <input type="text" id="project_id" name="project_id" placeholder="e.g., my-gcp-project-12345">
            <input type="submit" value="Login with Google & Generate Credential">
        </form>
    """
    return create_page("Gemini Credential Generator", content)


@app.post("/login", response_class=RedirectResponse)
async def login(project_id: str = Form("")):
    """Starts the OAuth2 flow, using a special state for discovery."""
    client_config = {
        "web": {
            "client_id": settings.CLIENT_ID,
            "client_secret": settings.CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    flow = Flow.from_client_config(
        client_config, scopes=SCOPES, redirect_uri=REDIRECT_URI
    )
    auth_flow_state["flow"] = flow

    # If the user doesn't provide a project_id, use a special marker
    # in the state to trigger discovery in the callback.
    state_value = project_id if project_id else "__DISCOVER__"

    authorization_url, state = flow.authorization_url(
        access_type="offline",
        prompt="select_account",
        include_granted_scopes="true",
        state=state_value,
    )
    return RedirectResponse(authorization_url)


@app.get("/oauth2callback", response_class=HTMLResponse)
async def oauth2callback(request: Request):
    """Handles the OAuth2 callback, gets user info, and saves the credential file."""
    returned_state = request.query_params.get("state", "")
    flow = auth_flow_state.get("flow")
    if not flow:
        return create_page(
            "Error",
            "<h1>Error</h1><p>Authentication flow not initiated. Please start at the main page.</p>",
        )

    try:
        await asyncio.to_thread(
            flow.fetch_token, authorization_response=str(request.url)
        )
        creds = flow.credentials

        async with httpx.AsyncClient() as client:
            userinfo_resp = await client.get(
                "https://www.googleapis.com/oauth2/v2/userinfo",
                headers={"Authorization": f"Bearer {creds.token}"},
            )
            userinfo_resp.raise_for_status()
            user_email = userinfo_resp.json().get("email", "unknown_email")

            final_project_id = ""
            # Check if we need to discover the project ID
            if returned_state == "__DISCOVER__":
                logger.info(
                    "No project ID specified, attempting to discover it from the API..."
                )
                discovery_payload = {"metadata": get_client_metadata()}
                discovery_headers = {
                    "Authorization": f"Bearer {creds.token}",
                    "User-Agent": get_user_agent(),
                    "Content-Type": "application/json",
                }

                logger.debug(
                    format_log(
                        "Sending project discovery request",
                        discovery_payload,
                        is_json=True,
                    )
                )

                resp = await client.post(
                    f"{settings.CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
                    data=json.dumps(discovery_payload),
                    headers=discovery_headers,
                )

                try:
                    response_json = resp.json()
                    logger.debug(
                        format_log(
                            f"Received project discovery response ({resp.status_code})",
                            response_json,
                            is_json=True,
                        )
                    )
                    resp.raise_for_status()
                    final_project_id = response_json.get(
                        "cloudaicompanionProject", "unknown_project"
                    )
                    logger.info(f"Discovered project ID: {final_project_id}")
                except (json.JSONDecodeError, httpx.HTTPStatusError) as e:
                    logger.error(
                        f"Failed to discover project ID. Response text: {resp.text}"
                    )
                    raise e
            else:
                # Use the project ID passed in the state
                final_project_id = returned_state
                logger.info(f"Using specified project ID: {final_project_id}")

        creds_data = {
            "client_id": settings.CLIENT_ID,
            "client_secret": settings.CLIENT_SECRET,
            "refresh_token": creds.refresh_token,
            "token_uri": "https://oauth2.googleapis.com/token",
            "project_id": final_project_id,
            "user_email": user_email,
        }

        safe_email = sanitize_for_filename(user_email)
        safe_project = sanitize_for_filename(final_project_id)
        file_name = f"oauth_creds_{safe_email}_{safe_project}.json"
        file_path = settings.PERSISTENT_STORAGE_PATH / file_name

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(creds_data, f, indent=2)

        content = f"""
            <h1>Credential Generated!</h1>
            <p>Credential for <strong>{user_email}</strong> and project <strong>{final_project_id}</strong> has been saved to:</p>
            <div class="code-block">
                <code id="file-path">{file_path}</code>
                <button class="copy-btn" onclick="copyToClipboard('file-path', this)">Copy</button>
            </div>
            <a href="/" class="btn">Generate Another Credential</a>
            <p class="footer">You can now close this window and stop the script (Ctrl+C).</p>
        """
        return create_page("Success", content)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error during OAuth callback: {e}", exc_info=True)
        content = f"""
        <h1>Error</h1>
        <p>An HTTP error occurred while communicating with Google's servers. Please see the details below:</p>
        <div class="code-block"><code>{e.response.text}</code></div>
        <a href="/" class="btn">Try Again</a>
        """
        return create_page("Error", content)
    except Exception as e:
        logger.error(f"Error during OAuth callback: {e}", exc_info=True)
        content = f"""
        <h1>Error</h1>
        <p>An unexpected error occurred while generating the credential. Please see the details below:</p>
        <div class="code-block"><code>{e}</code></div>
        <a href="/" class="btn">Try Again</a>
        """
        return create_page("Error", content)

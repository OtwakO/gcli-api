import base64
import json
import logging
import os
import time
from datetime import datetime, timezone
import asyncio
import httpx

from fastapi import Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow

from .constants import CODE_ASSIST_ENDPOINT, SCOPES
from .settings import settings
from .utils import get_client_metadata, get_user_agent

# -- OAuth2 Patches --
os.environ["OAUTHLIB_RELAX_TOKEN_SCOPE"] = "1"
os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# --- Global State ---
credentials = None
user_project_id = None
onboarding_complete = False
auth_flow_state = {"flow": None}
credentials_from_env = False  # Track if credentials came from environment variable


def authenticate_user(request: Request):
    """Authenticate the user. Allows anonymous access to auth endpoints."""
    if request.url.path in ["/login", "/oauth2callback", "/"]:
        return "anonymous_oauth_user"

    auth_header = request.headers.get("authorization", "")
    api_key_query = request.query_params.get("key")
    goog_api_key_header = request.headers.get("x-goog-api-key")

    if (
        (api_key_query and api_key_query == settings.GEMINI_AUTH_PASSWORD)
        or (
            goog_api_key_header and goog_api_key_header == settings.GEMINI_AUTH_PASSWORD
        )
        or (
            auth_header.startswith("Bearer ")
            and auth_header[7:] == settings.GEMINI_AUTH_PASSWORD
        )
    ):
        return "api_key_user"

    if auth_header.startswith("Basic "):
        try:
            decoded = base64.b64decode(auth_header[6:]).decode("utf-8")
            username, password = decoded.split(":", 1)
            if password == settings.GEMINI_AUTH_PASSWORD:
                return username
        except Exception:
            pass

    raise HTTPException(
        status_code=401,
        detail="Invalid authentication credentials.",
        headers={"WWW-Authenticate": "Basic"},
    )


def save_credentials(creds, project_id=None):
    """Saves credentials and project_id to the persistent file."""
    global credentials_from_env

    # Don't save credentials to file if they came from environment variable,
    # but still save project_id if provided and no file exists or file lacks project_id
    if credentials_from_env:
        if project_id and os.path.exists(settings.CREDENTIAL_FILE):
            try:
                with open(settings.CREDENTIAL_FILE, "r") as f:
                    existing_data = json.load(f)
                # Only update project_id if it's missing from the file
                if "project_id" not in existing_data:
                    existing_data["project_id"] = project_id
                    with open(settings.CREDENTIAL_FILE, "w") as f:
                        json.dump(existing_data, f, indent=2)
                    logging.info(
                        f"Added project_id {project_id} to existing credential file"
                    )
            except Exception as e:
                logging.warning(f"Could not update project_id in credential file: {e}")
        return

    creds_data = {
        "client_id": settings.CLIENT_ID,
        "client_secret": settings.CLIENT_SECRET,
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "scopes": creds.scopes if creds.scopes else SCOPES,
        "token_uri": "https://oauth2.googleapis.com/token",
    }

    if creds.expiry:
        if creds.expiry.tzinfo is None:
            expiry_utc = creds.expiry.replace(tzinfo=timezone.utc)
        else:
            expiry_utc = creds.expiry
        creds_data["expiry"] = expiry_utc.isoformat()

    if project_id:
        creds_data["project_id"] = project_id
    elif os.path.exists(settings.CREDENTIAL_FILE):
        try:
            with open(settings.CREDENTIAL_FILE, "r") as f:
                existing_data = json.load(f)
                if "project_id" in existing_data:
                    creds_data["project_id"] = existing_data["project_id"]
        except Exception:
            pass

    with open(settings.CREDENTIAL_FILE, "w") as f:
        json.dump(creds_data, f, indent=2)
    logging.info(f"Credentials saved to {settings.CREDENTIAL_FILE}")


def _parse_creds_data(raw_creds_data: dict) -> dict:
    creds_data = raw_creds_data.copy()
    if "access_token" in creds_data and "token" not in creds_data:
        creds_data["token"] = creds_data["access_token"]
    if "scope" in creds_data and "scopes" not in creds_data:
        creds_data["scopes"] = creds_data["scope"].split()
    if "expiry" in creds_data:
        expiry_str = creds_data["expiry"]
        if isinstance(expiry_str, str) and (
            "+00:00" in expiry_str or "Z" in expiry_str
        ):
            try:
                if "+00:00" in expiry_str:
                    parsed_expiry = datetime.fromisoformat(expiry_str)
                elif expiry_str.endswith("Z"):
                    parsed_expiry = datetime.fromisoformat(
                        expiry_str.replace("Z", "+00:00")
                    )
                else:
                    parsed_expiry = datetime.fromisoformat(expiry_str)
                timestamp = parsed_expiry.timestamp()
                creds_data["expiry"] = datetime.utcfromtimestamp(timestamp).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                )
                logging.info(
                    f"Converted expiry format from '{expiry_str}' to '{creds_data['expiry']}'"
                )
            except Exception as expiry_error:
                logging.warning(
                    f"Could not parse expiry format '{expiry_str}': {expiry_error}, removing expiry field"
                )
                del creds_data["expiry"]
    return creds_data


async def _load_creds_from_env():
    global credentials, credentials_from_env, user_project_id
    env_creds_json = settings.OAUTH_CREDS_JSON
    if not env_creds_json:
        return None
    try:
        raw_env_creds_data = json.loads(env_creds_json)
        if (
            "refresh_token" in raw_env_creds_data
            and raw_env_creds_data["refresh_token"]
        ):
            logging.info(
                "Environment refresh token found - ensuring credentials load successfully"
            )
            try:
                creds_data = _parse_creds_data(raw_env_creds_data)
                credentials = Credentials.from_authorized_user_info(creds_data, SCOPES)
                credentials_from_env = True
                if "project_id" in raw_env_creds_data:
                    user_project_id = raw_env_creds_data["project_id"]
                    logging.info(
                        f"Extracted project_id from environment credentials: {user_project_id}"
                    )
                if credentials.expired and credentials.refresh_token:
                    logging.info(
                        "Environment credentials expired, attempting refresh..."
                    )
                    await asyncio.to_thread(credentials.refresh, GoogleAuthRequest())
                    logging.info("Environment credentials refreshed successfully")
                elif not credentials.expired:
                    logging.info(
                        "Environment credentials are still valid, no refresh needed"
                    )
                elif not credentials.refresh_token:
                    logging.warning(
                        "Environment credentials expired but no refresh token available"
                    )
                return credentials
            except Exception as parsing_error:
                logging.warning(
                    f"Failed to parse environment credentials normally: {parsing_error}"
                )
                logging.info(
                    "Attempting to create minimal environment credentials with refresh token"
                )
                try:
                    minimal_creds_data = {
                        "client_id": raw_env_creds_data.get(
                            "client_id", settings.CLIENT_ID
                        ),
                        "client_secret": raw_env_creds_data.get(
                            "client_secret", settings.CLIENT_SECRET
                        ),
                        "refresh_token": raw_env_creds_data["refresh_token"],
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                    credentials = Credentials.from_authorized_user_info(
                        minimal_creds_data, SCOPES
                    )
                    credentials_from_env = True
                    if "project_id" in raw_env_creds_data:
                        user_project_id = raw_env_creds_data["project_id"]
                        logging.info(
                            f"Extracted project_id from minimal environment credentials: {user_project_id}"
                        )
                    logging.info("Refreshing minimal environment credentials...")
                    await asyncio.to_thread(credentials.refresh, GoogleAuthRequest())
                    logging.info(
                        "Minimal environment credentials refreshed successfully"
                    )
                    return credentials
                except Exception as minimal_error:
                    logging.error(
                        f"Failed to create minimal environment credentials: {minimal_error}"
                    )
        else:
            logging.warning("No refresh token found in environment credentials")
    except Exception as e:
        logging.error(f"Failed to parse environment credentials JSON: {e}")
    return None


async def _load_creds_from_file():
    global credentials, credentials_from_env, user_project_id
    if not os.path.exists(settings.CREDENTIAL_FILE):
        return None
    try:
        with open(settings.CREDENTIAL_FILE, "r") as f:
            raw_creds_data = json.load(f)
        if "refresh_token" in raw_creds_data and raw_creds_data["refresh_token"]:
            logging.info("Refresh token found - ensuring credentials load successfully")
            try:
                creds_data = _parse_creds_data(raw_creds_data)
                credentials = Credentials.from_authorized_user_info(creds_data, SCOPES)
                credentials_from_env = bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
                if credentials.expired and credentials.refresh_token:
                    logging.info(
                        "File-based credentials expired, attempting refresh..."
                    )
                    await asyncio.to_thread(credentials.refresh, GoogleAuthRequest())
                    logging.info("File-based credentials refreshed successfully")
                    save_credentials(credentials)
                elif not credentials.expired:
                    logging.info(
                        "File-based credentials are still valid, no refresh needed"
                    )
                elif not credentials.refresh_token:
                    logging.warning(
                        "File-based credentials expired but no refresh token available"
                    )
                return credentials
            except Exception as parsing_error:
                logging.warning(
                    f"Failed to parse credentials normally: {parsing_error}"
                )
                logging.info(
                    "Attempting to create minimal credentials with refresh token"
                )
                try:
                    minimal_creds_data = {
                        "client_id": raw_creds_data.get(
                            "client_id", settings.CLIENT_ID
                        ),
                        "client_secret": raw_creds_data.get(
                            "client_secret", settings.CLIENT_SECRET
                        ),
                        "refresh_token": raw_creds_data["refresh_token"],
                        "token_uri": "https://oauth2.googleapis.com/token",
                    }
                    credentials = Credentials.from_authorized_user_info(
                        minimal_creds_data, SCOPES
                    )
                    credentials_from_env = bool(
                        os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                    )
                    logging.info("Refreshing minimal credentials...")
                    await asyncio.to_thread(credentials.refresh, GoogleAuthRequest())
                    logging.info("Minimal credentials refreshed successfully")
                    save_credentials(credentials)
                    return credentials
                except Exception as minimal_error:
                    logging.error(
                        f"Failed to create minimal credentials: {minimal_error}"
                    )
        else:
            logging.warning("No refresh token found in credentials file")
    except Exception as e:
        logging.error(
            f"Failed to read credentials file {settings.CREDENTIAL_FILE}: {e}"
        )
    return None


async def get_credentials():
    """Loads credentials matching gemini-cli OAuth2 flow."""
    global credentials
    if credentials and credentials.token:
        return credentials

    # Try loading from environment variable first
    creds = await _load_creds_from_env()
    if creds:
        return creds

    # Then try loading from file
    creds = await _load_creds_from_file()
    if creds:
        return creds

    return None


async def onboard_user(creds, project_id):
    """Ensures the user is onboarded."""
    global onboarding_complete
    if onboarding_complete:
        return

    if creds.expired and creds.refresh_token:
        await asyncio.to_thread(creds.refresh, GoogleAuthRequest())
        save_credentials(creds)

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }
    load_assist_payload = {
        "cloudaicompanionProject": project_id,
        "metadata": get_client_metadata(project_id),
    }

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
                data=json.dumps(load_assist_payload, ensure_ascii=False),
                headers=headers,
            )
            resp.raise_for_status()
            load_data = resp.json()

            if load_data.get("currentTier"):
                onboarding_complete = True
                return

            tier = next(
                (t for t in load_data.get("allowedTiers", []) if t.get("isDefault")),
                {"id": "legacy-tier", "userDefinedCloudaicompanionProject": True},
            )

            if tier.get("userDefinedCloudaicompanionProject") and not project_id:
                raise ValueError(
                    "This account requires setting the GOOGLE_CLOUD_PROJECT env var."
                )

            onboard_req_payload = {
                "tierId": tier.get("id"),
                "cloudaicompanionProject": project_id,
                "metadata": get_client_metadata(project_id),
            }

            while True:
                onboard_resp = await client.post(
                    f"{CODE_ASSIST_ENDPOINT}/v1internal:onboardUser",
                    data=json.dumps(onboard_req_payload, ensure_ascii=False),
                    headers=headers,
                )
                onboard_resp.raise_for_status()
                lro_data = onboard_resp.json()
                if lro_data.get("done"):
                    onboarding_complete = True
                    break
                await asyncio.sleep(5)
    except httpx.HTTPStatusError as e:
        logging.error(f"Error during onboarding: {e.response.text}")
        raise


async def get_user_project_id(creds):
    """Gets project ID with robust logic."""
    global user_project_id
    if user_project_id:
        return user_project_id

    env_project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if env_project_id:
        user_project_id = env_project_id
        save_credentials(creds, user_project_id)
        return user_project_id

    if os.path.exists(settings.CREDENTIAL_FILE):
        try:
            with open(settings.CREDENTIAL_FILE, "r") as f:
                cached_project_id = json.load(f).get("project_id")
                if cached_project_id:
                    user_project_id = cached_project_id
                    return user_project_id
        except Exception as e:
            logging.warning(f"Could not load project ID from cache: {e}")

    if creds.expired and creds.refresh_token:
        await asyncio.to_thread(creds.refresh, GoogleAuthRequest())
        save_credentials(creds)

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist",
            data=json.dumps({"metadata": get_client_metadata()}, ensure_ascii=False),
            headers=headers,
        )
        resp.raise_for_status()
        data = resp.json()
        api_project_id = data.get("cloudaicompanionProject")
        if not api_project_id:
            raise ValueError("Could not find 'cloudaicompanionProject' from API response.")

        user_project_id = api_project_id
        save_credentials(creds, user_project_id)
        return user_project_id


async def validate_project_access(creds, project_id):
    """Checks if the user has access to the specified project."""
    if not project_id:
        return True

    if creds.expired and creds.refresh_token:
        await asyncio.to_thread(creds.refresh, GoogleAuthRequest())

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json",
        "User-Agent": get_user_agent(),
    }
    validation_url = f"https://cloudresourcemanager.googleapis.com/v1/projects/{project_id}"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(validation_url, headers=headers)
            resp.raise_for_status()
        logging.info(f"Successfully validated user access to project {project_id}.")
        return True
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 403:
            logging.error(f"Validation failed: User does not have access to project {project_id}.")
            return False
        else:
            logging.error(f"An unexpected error occurred during project validation: {e.response.text}")
            raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during project validation: {e}")
        raise


async def login():
    client_config = {
        "web": {
            "client_id": settings.CLIENT_ID,
            "client_secret": settings.CLIENT_SECRET,
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
        }
    }
    redirect_uri = f"{settings.DOMAIN_NAME}/oauth2callback"

    flow = Flow.from_client_config(
        client_config, scopes=SCOPES, redirect_uri=redirect_uri
    )
    auth_flow_state["flow"] = flow
    authorization_url, _ = flow.authorization_url(
        access_type="offline", prompt="consent", include_granted_scopes="true"
    )

    return RedirectResponse(authorization_url)


async def oauth2callback(request: Request):
    global credentials, user_project_id, onboarding_complete
    flow = auth_flow_state.get("flow")
    if not flow:
        raise HTTPException(
            status_code=400, detail="Auth flow not initiated. Visit /login."
        )

    try:
        await asyncio.to_thread(flow.fetch_token, authorization_response=str(request.url))
        credentials = flow.credentials

        specified_project_id = settings.GOOGLE_CLOUD_PROJECT
        if specified_project_id:
            logging.info(f"GOOGLE_CLOUD_PROJECT is set to '{specified_project_id}'. Validating access...")
            access_granted = await validate_project_access(credentials, specified_project_id)
            if not access_granted:
                credentials = None
                return Response(
                    content=f"<h1>Authentication Failed</h1><p>The logged-in user does not have access to the specified Google Cloud Project: <strong>{specified_project_id}</strong>. Please log in with a different account or check your project permissions.</p>",
                    media_type="text/html",
                    status_code=403
                )
            proj_id = specified_project_id
        else:
            proj_id = await get_user_project_id(credentials)

        save_credentials(credentials, proj_id)

        user_project_id = None
        onboarding_complete = False
        
        if proj_id:
            await onboard_user(credentials, proj_id)

        return Response(
            content="<h1>Authentication Successful!</h1><p>You can close this window.</p>",
            media_type="text/html",
        )
    except Exception as e:
        logging.error(f"Error during OAuth callback: {e}")
        credentials = None
        raise HTTPException(status_code=500, detail=f"Failed to fetch OAuth token or validate project: {e}")

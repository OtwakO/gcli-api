import asyncio
import json
import random

from fastapi import HTTPException

from ..core.upstream_auth import OAuthStrategy
from ..core.credential_manager import ManagedCredential
from ..core.google_api_client import send_request
from ..core.settings import settings
from ..utils.logger import get_logger
from ..utils.utils import get_client_metadata

logger = get_logger(__name__)


class OnboardingService:
    """Handles the complex logic of user onboarding and credential preparation."""

    async def prepare_credential(self, managed_cred: ManagedCredential) -> str:
        """
        Ensures the credential has a project ID and the user is onboarded.
        Returns the project ID.
        """
        project_id = managed_cred.project_id
        if not project_id:
            project_id = await self._fetch_project_id(managed_cred)

        if not managed_cred.is_onboarded:
            await self._perform_onboarding(managed_cred, project_id)

        return project_id

    async def _fetch_project_id(self, managed_cred: ManagedCredential) -> str:
        """Gets project ID for the given credential, using cache if available."""
        if managed_cred.project_id:
            return managed_cred.project_id

        try:
            auth_strategy = OAuthStrategy(managed_cred)
            target_url = f"{settings.CODE_ASSIST_ENDPOINT}/v1internal:loadCodeAssist"
            payload = {"metadata": get_client_metadata()}

            resp = await send_request(target_url, payload, auth_strategy)
            data = resp.json()

            api_project_id = data.get("cloudaicompanionProject")
            if not api_project_id:
                raise ValueError(
                    "Could not find 'cloudaicompanionProject' from API response."
                )

            managed_cred.project_id = api_project_id
            logger.info(
                f"Discovered project ID for {managed_cred.user_email}: {api_project_id}"
            )
            return api_project_id
        except (ValueError, json.JSONDecodeError, HTTPException) as e:
            logger.error(
                f"Could not discover project ID from API for {managed_cred.user_email}: {e}"
            )
            raise HTTPException(
                status_code=500,
                detail="Failed to discover user project ID from Google API.",
            )

    async def _perform_onboarding(
        self, managed_cred: ManagedCredential, project_id: str
    ):
        """Ensures the user associated with the credential is onboarded."""
        if managed_cred.is_onboarded:
            return

        try:
            auth_strategy = OAuthStrategy(managed_cred)
            base_url = f"{settings.CODE_ASSIST_ENDPOINT}/v1internal"

            # Step 1: Load Code Assist to check current status
            load_assist_payload = {
                "cloudaicompanionProject": project_id,
                "metadata": get_client_metadata(project_id),
            }
            resp = await send_request(
                f"{base_url}:loadCodeAssist", load_assist_payload, auth_strategy
            )
            load_data = resp.json()

            if load_data.get("currentTier"):
                managed_cred.is_onboarded = True
                logger.info(
                    f"User {managed_cred.user_email} is already onboarded for project {project_id}."
                )
                return

            # Step 2: Find the default tier and onboard the user
            tier = next(
                (t for t in load_data.get("allowedTiers", []) if t.get("isDefault")),
                {"id": "legacy-tier"},
            )

            onboard_req_payload = {
                "tierId": tier.get("id"),
                "cloudaicompanionProject": project_id,
                "metadata": get_client_metadata(project_id),
            }

            logger.info(
                f"Onboarding user {managed_cred.user_email} for project {project_id}..."
            )
            max_attempts = 5
            base_delay = 1.0
            for attempt in range(max_attempts):
                onboard_resp = await send_request(
                    f"{base_url}:onboardUser", onboard_req_payload, auth_strategy
                )
                lro_data = onboard_resp.json()
                if lro_data.get("done"):
                    managed_cred.is_onboarded = True
                    logger.info(
                        f"Successfully onboarded user {managed_cred.user_email}."
                    )
                    return

                delay = (base_delay * 2**attempt) + (random.uniform(0, 1))
                logger.info(
                    f"Onboarding not complete, retrying in {delay:.2f} seconds..."
                )
                await asyncio.sleep(delay)

            logger.warning(
                f"Onboarding LRO for {managed_cred.user_email} did not complete after {max_attempts} attempts."
            )

        except HTTPException as e:
            logger.error(
                f"Error during onboarding for project {project_id}: {getattr(e, 'detail', e)}"
            )
            raise HTTPException(
                status_code=e.status_code,
                detail=f"Onboarding failed: {getattr(e, 'detail', e)}",
            )


onboarding_service = OnboardingService()

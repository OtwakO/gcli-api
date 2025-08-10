import json
from typing import Dict

from fastapi import HTTPException
from pydantic import ValidationError

from ..core.upstream_auth import OAuthStrategy
from ..core.credential_manager import ManagedCredential
from ..core.google_api_client import send_request
from ..models.gemini import CountTokensResponse
from ..services.onboarding_service import onboarding_service
from ..utils.logger import get_logger
from ..utils.utils import build_gemini_url

logger = get_logger(__name__)


class ModelService:
    """
    A service for handling model-specific actions like counting tokens.
    """

    async def count_tokens(
        self,
        model_name: str,
        managed_cred: ManagedCredential,
        payload: Dict,
    ) -> CountTokensResponse:
        """
        Handles the logic for a countTokens request.
        """
        try:
            # Ensure the credential is fully prepared (onboarded, has project ID)
            await onboarding_service.prepare_credential(managed_cred)
            target_url = build_gemini_url("countTokens", model_name)

            # The countTokens API has a specific payload structure
            final_payload = {"request": {"model": model_name, **payload}}
            auth_strategy = OAuthStrategy(managed_cred)

            upstream_response = await send_request(
                target_url, final_payload, auth_strategy
            )

            validated_response = CountTokensResponse.model_validate(
                upstream_response.json()
            )
            return validated_response

        except (ValidationError, json.JSONDecodeError) as e:
            logger.error(f"Error processing countTokens response: {e}", exc_info=True)
            # Re-raising as HTTPException to be caught by the global handler
            raise HTTPException(
                status_code=500, detail=f"Error processing countTokens response: {e}"
            )
        except HTTPException:
            # Re-raise exceptions that are already structured for HTTP response
            raise
        except Exception as e:
            logger.error(
                f"An unexpected error occurred in countTokens service: {e}",
                exc_info=True,
            )
            raise HTTPException(status_code=500, detail=f"Internal server error: {e}")


model_service = ModelService()

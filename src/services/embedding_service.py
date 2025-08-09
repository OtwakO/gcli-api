from typing import Any, Dict, Union

from fastapi import HTTPException
from pydantic import ValidationError

from ..core.google_api_client import send_public_api_request
from ..core.settings import settings
from ..models.gemini import BatchEmbedContentResponse, EmbedContentResponse
from ..utils.logger import get_logger
from ..utils.utils import build_gemini_url

logger = get_logger(__name__)


class EmbeddingService:
    """
    A service dedicated to handling requests to the public Gemini Embedding API.
    """

    async def execute_embedding_request(
        self,
        action: str,
        model_name: str,
        payload: Dict[str, Any],
    ) -> Union[EmbedContentResponse, BatchEmbedContentResponse]:
        """
        Executes a request against the public embedding API endpoint using an API key.

        Args:
            action: The API action to perform (e.g., "embedContent").
            model_name: The name of the embedding model.
            payload: The request payload.

        Returns:
            A validated Pydantic model of the response.

        Raises:
            HTTPException: If the API call fails or the response is invalid.
        """
        if not settings.EMBEDDING_GEMINI_API_KEY:
            logger.error(
                "Embedding request failed: EMBEDDING_GEMINI_API_KEY is not configured."
            )
            raise HTTPException(
                status_code=500,
                detail="The server is not configured to handle embedding requests. "
                "The EMBEDDING_GEMINI_API_KEY is missing.",
            )

        try:
            target_url = build_gemini_url(action, model_name)

            upstream_response = await send_public_api_request(
                target_url, settings.EMBEDDING_GEMINI_API_KEY, payload
            )
            gemini_response_obj = upstream_response.json()

            if action == "batchEmbedContents":
                return BatchEmbedContentResponse.model_validate(gemini_response_obj)
            else:  # embedContent
                return EmbedContentResponse.model_validate(gemini_response_obj)

        except (HTTPException, ValidationError) as e:
            # Re-raise known exceptions to be handled by the global error handler.
            raise e
        except Exception as e:
            logger.error(
                f"Error processing public embedding request for action '{action}': {e}",
                exc_info=True,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error during embedding request: {e}",
            )


embedding_service = EmbeddingService()

from typing import Any

from pydantic import BaseModel as PydanticBaseModel

from ..utils.logger import get_logger

logger = get_logger(__name__)


class LoggingBaseModel(PydanticBaseModel):
    """
    A custom Pydantic BaseModel that allows extra fields and logs a warning
    when they are encountered.
    """

    class Config:
        extra = "allow"

    def __init__(self, **data: Any):
        super().__init__(**data)
        if hasattr(self, "__pydantic_extra__") and self.__pydantic_extra__:
            logger.warning(
                f"Request model '{type(self).__name__}' received with undefined parameters. "
                f"These will be passed through. Consider updating the model definition. "
                f"Extra fields: {self.__pydantic_extra__}"
            )

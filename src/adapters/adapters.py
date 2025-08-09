from dataclasses import dataclass
from typing import Any, Callable, Dict, Type, Tuple

from . import claude_transformers, openai_transformers
from .formatters import (
    ClaudeFormatter,
    Formatter,
    OpenAIEmbeddingFormatter,
    OpenAIFormatter,
)
from ..models.gemini import GeminiRequest

# --- Type Aliases for Transformers ---

# For chat completion requests
ChatRequestTransformer = Callable[[Any], Tuple[str, GeminiRequest]]

# For embedding requests, which also determines the action (e.g., embedContent)
EmbeddingRequestTransformer = Callable[[Any], Tuple[str, str, Dict[str, Any]]]


# --- Adapter Dataclasses ---


@dataclass
class ApiAdapter:
    """Groups the components needed to adapt a vendor API for chat completions."""

    request_transformer: ChatRequestTransformer
    formatter_class: Type[Formatter]


@dataclass
class EmbeddingAdapter:
    """Groups the components needed to adapt a vendor API for embeddings."""

    request_transformer: EmbeddingRequestTransformer
    formatter_class: Type[Formatter]


# --- Adapter Instances ---


def _openai_transform_request(req: Any) -> Tuple[str, GeminiRequest]:
    """Wrapper to make openai_request_to_gemini match the expected signature."""
    return req.model, openai_transformers.openai_request_to_gemini(req)


# Adapter for OpenAI Chat Completions
openai_adapter = ApiAdapter(
    request_transformer=_openai_transform_request,
    formatter_class=OpenAIFormatter,
)

# Adapter for OpenAI Embeddings
openai_embedding_adapter = EmbeddingAdapter(
    request_transformer=openai_transformers.openai_embedding_request_transformer,
    formatter_class=OpenAIEmbeddingFormatter,
)

# Adapter for Claude API
claude_adapter = ApiAdapter(
    request_transformer=claude_transformers.claude_request_to_gemini,
    formatter_class=ClaudeFormatter,
)

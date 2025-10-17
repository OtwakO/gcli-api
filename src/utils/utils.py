import copy
import platform
import uuid
from typing import Any, Dict, List, Optional

from ..core.settings import settings
from .constants import CLI_VERSION


def build_gemini_url(action: str, model_name: str = "") -> str:
    """
    Constructs the full URL for a given Gemini API action, directing the request
    to the appropriate endpoint based on the action.
    """
    embedding_actions = ["embedContent", "batchEmbedContents"]

    if action in embedding_actions:
        # Embedding actions are routed to the public Gemini API endpoint.
        base_url = settings.GEMINI_PUBLIC_ENDPOINT
        return f"{base_url}/v1beta/models/{model_name}:{action}"
    else:
        # All other actions are routed to the Cloud Code Assist endpoint.
        url = f"{settings.CODE_ASSIST_ENDPOINT}/v1internal:{action}"
        if "stream" in action.lower():
            url += "?alt=sse"
        return url


def generate_response_id(prefix: str) -> str:
    """Generates a unique response ID with a given prefix."""
    if prefix == "msg":
        return f"msg_{uuid.uuid4().hex}"
    return f"{prefix}-{uuid.uuid4()}"


def get_user_agent():
    """Generate User-Agent string matching gemini-cli format."""
    version = CLI_VERSION
    system = platform.system()
    arch = platform.machine()
    return f"GeminiCLI/{version} ({system}; {arch})"


def get_platform_string():
    """Generate platform string matching gemini-cli format."""
    system = platform.system().upper()
    arch = platform.machine().upper()

    # Map to gemini-cli platform format
    if system == "DARWIN":
        if arch in ["ARM64", "AARCH64"]:
            return "DARWIN_ARM64"
        else:
            return "DARWIN_AMD64"
    elif system == "LINUX":
        if arch in ["ARM64", "AARCH64"]:
            return "LINUX_ARM64"
        else:
            return "LINUX_AMD64"
    elif system == "WINDOWS":
        return "WINDOWS_AMD64"
    else:
        return "PLATFORM_UNSPECIFIED"


def get_client_metadata(project_id=None):
    return {
        "ideType": "IDE_UNSPECIFIED",
        "platform": get_platform_string(),
        "pluginType": "GEMINI",
        "duetProject": project_id,
    }


def _redact_recursive(obj: Any):
    """Recursively traverses a dictionary or list and redacts sensitive keys."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in ["text", "data"]:
                obj[key] = "<REDACTED>"
            else:
                _redact_recursive(value)
    elif isinstance(obj, list):
        for item in obj:
            _redact_recursive(item)


def create_redacted_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep copies and redacts sensitive fields from a payload for safe logging.
    Redacts 'text' and 'data' fields wherever they appear in the structure.
    """
    if not payload:
        return {}

    payload_copy = copy.deepcopy(payload)
    _redact_recursive(payload_copy)
    return payload_copy


def summarize_embedding_logs(log_data: Any) -> Any:
    """
    Recursively traverses a dictionary or list and replaces embedding
    value lists with a summary string for cleaner logging.
    Operates on a deep copy to avoid side effects.
    """
    if not isinstance(log_data, (dict, list)):
        return log_data

    log_data_copy = copy.deepcopy(log_data)

    def _summarize_recursive(obj: Any):
        if isinstance(obj, dict):
            # Handle single embedding: { "embedding": { "values": [...] } }
            if "embedding" in obj and isinstance(obj["embedding"], dict):
                if "values" in obj["embedding"] and isinstance(
                    obj["embedding"]["values"], list
                ):
                    count = len(obj["embedding"]["values"])
                    obj["embedding"]["values"] = f"<{count} embedding values>"

            # Handle batch embeddings: { "embeddings": [ { "values": [...] } ] }
            if "embeddings" in obj and isinstance(obj["embeddings"], list):
                for item in obj["embeddings"]:
                    if (
                        isinstance(item, dict)
                        and "values" in item
                        and isinstance(item["values"], list)
                    ):
                        count = len(item["values"])
                        item["values"] = f"<{count} embedding values>"

            # Recurse into other dictionary values
            for value in obj.values():
                _summarize_recursive(value)

        elif isinstance(obj, list):
            for item in obj:
                _summarize_recursive(item)

    _summarize_recursive(log_data_copy)
    return log_data_copy


def get_extra_fields(model) -> Dict[str, Any]:
    """Safely extracts the dictionary of extra fields from a Pydantic model instance."""
    if hasattr(model, "__pydantic_extra__") and model.__pydantic_extra__:
        return model.__pydantic_extra__
    return {}


def dump_model_with_extras(model, **kwargs) -> Dict[str, Any]:
    """Dumps a Pydantic model to a dict, including any extra fields."""
    data = model.model_dump(**kwargs)
    data.update(get_extra_fields(model))
    return data


def sanitize_gemini_tools(
    tools: Optional[List[Dict[str, Any]]],
) -> Optional[List[Dict[str, Any]]]:
    """
    Recursively removes unsupported keys from Gemini tool definitions based on a
    centralized configuration in `settings.UNSUPPORTED_TOOL_SCHEMA_KEYS`.

    This utility ensures that tool schemas sent to the Gemini API are compliant
    by stripping out fields that would otherwise cause validation errors.

    Args:
        tools: A Gemini-formatted tools list (e.g., [{'functionDeclarations': ...}]).

    Returns:
        A sanitized deep copy of the tools list, or None if input is None.
    """
    if not tools:
        return None

    tools_copy = copy.deepcopy(tools)
    unsupported_keys = set(settings.UNSUPPORTED_TOOL_SCHEMA_KEYS)

    def _recursive_remove_keys(obj: Any):
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                if key in unsupported_keys:
                    del obj[key]
                else:
                    _recursive_remove_keys(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                _recursive_remove_keys(item)

    if (
        tools_copy
        and isinstance(tools_copy, list)
        and tools_copy[0].get("functionDeclarations")
    ):
        for func_dec in tools_copy[0]["functionDeclarations"]:
            if "parameters" in func_dec:
                _recursive_remove_keys(func_dec["parameters"])

    return tools_copy

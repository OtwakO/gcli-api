import platform
import copy
from typing import Any, Dict

from .constants import CLI_VERSION
from ..core.settings import settings


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

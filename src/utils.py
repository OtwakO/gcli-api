import platform
import copy
from typing import Any, Dict

from .constants import CLI_VERSION


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
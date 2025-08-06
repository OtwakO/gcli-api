import logging
import json
from rich.logging import RichHandler
from typing import Any

from .settings import settings

def setup_logging():
    """
    Configures the logging for the entire application, using RichHandler for
    pretty console output.
    """
    log_level = "DEBUG" if settings.DEBUG else "INFO"
    
    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove any existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # Create a RichHandler for console output
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_time=True,
        show_level=True,
        show_path=False,
        log_time_format="[%Y-%m-%d %H:%M:%S]",
    )
    
    # Create a formatter and set it for the handler
    formatter = logging.Formatter(
        fmt="%(message)s",
        datefmt="[%X]",
    )
    rich_handler.setFormatter(formatter)

    # Add the handler to the root logger
    root_logger.addHandler(rich_handler)

    # Silence overly verbose libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance for the given name.
    """
    return logging.getLogger(name)

def format_log(title: str, content: Any, is_json: bool = False, indent: int = 2) -> str:
    """
    Formats a log message with a title and structured content.
    
    Args:
        title: The title for the log section.
        content: The content to be logged (can be a dict, str, or other object).
        is_json: If True, tries to format the content as a JSON string.
        indent: The indentation level for JSON content.
        
    Returns:
        A formatted string ready for logging.
    """
    formatted_content = content
    if is_json:
        try:
            if isinstance(content, str):
                # If content is already a JSON string, parse and re-dump for pretty printing
                content = json.loads(content)
            formatted_content = json.dumps(content, indent=indent, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            # Fallback for non-JSON or non-serializable content
            pass

    return f"---" + " " + title + " ---" + "\n" + str(formatted_content)

# Initialize logging when the module is imported
setup_logging()

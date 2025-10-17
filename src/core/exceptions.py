class UpstreamResponseError(Exception):
    """Base exception for issues with upstream API responses."""

    pass


class MalformedContentError(UpstreamResponseError):
    """
    Raised when the upstream API returns a response that is structurally valid
    but contains no usable content (e.g., only candidates with empty 'content' blocks).
    """

    def __init__(self, finish_reason: str | None = None):
        message = "Upstream API returned a response with no valid content."
        if finish_reason:
            message += f" Finish Reason: {finish_reason}"

        self.message = message
        self.finish_reason = finish_reason
        super().__init__(self.message)


class UpstreamHttpError(UpstreamResponseError):
    """Raised for any handled HTTP error from the upstream API."""

    def __init__(self, status_code: int, detail: str | dict):
        self.status_code = status_code
        self.detail = detail
        message = f"Upstream API returned an HTTP {status_code} error."
        super().__init__(message)

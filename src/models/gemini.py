from typing import Any, Dict, List, Optional, Union

from pydantic import field_validator

from .base import LoggingBaseModel


# Gemini Models
class GeminiFunctionCall(LoggingBaseModel):
    name: str
    args: Dict[str, Any]


class GeminiFunctionResponse(LoggingBaseModel):
    name: str
    response: Dict[str, Any]


class GeminiFileData(LoggingBaseModel):
    mimeType: str
    fileUri: str


class GeminiInlineData(LoggingBaseModel):
    mimeType: str
    data: str


class GeminiExecutableCode(LoggingBaseModel):
    language: str
    code: str


class GeminiCodeExecutionResult(LoggingBaseModel):
    outcome: str
    output: str


class GeminiPart(LoggingBaseModel):
    text: Optional[str] = None
    inlineData: Optional[GeminiInlineData] = None
    functionCall: Optional[GeminiFunctionCall] = None
    functionResponse: Optional[GeminiFunctionResponse] = None
    fileData: Optional[GeminiFileData] = None
    executableCode: Optional[GeminiExecutableCode] = None
    codeExecutionResult: Optional[GeminiCodeExecutionResult] = None
    thought: Optional[bool] = None
    thoughtSignature: Optional[str] = None


class GeminiContent(LoggingBaseModel):
    role: str
    parts: List[GeminiPart] = []


class GeminiSystemInstruction(LoggingBaseModel):
    parts: List[GeminiPart]


class GeminiFunctionCallingConfig(LoggingBaseModel):
    mode: Optional[str] = None  # "AUTO", "ANY", "NONE"
    allowedFunctionNames: Optional[List[str]] = None


class GeminiToolConfig(LoggingBaseModel):
    functionCallingConfig: Optional[GeminiFunctionCallingConfig] = None


class GeminiRequest(LoggingBaseModel):
    contents: List[GeminiContent]
    systemInstruction: Optional[GeminiSystemInstruction] = None
    tools: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None
    toolConfig: Optional[GeminiToolConfig] = None
    safetySettings: Optional[List[Dict[str, Any]]] = None
    generationConfig: Optional[Dict[str, Any]] = None
    cachedContent: Optional[str] = None

    @field_validator("tools")
    @classmethod
    def normalize_tools_format(cls, v):
        if v is None:
            return None
        # If v is a dict like {"functionDeclarations": [...]}, wrap it in a list.
        if isinstance(v, dict) and "functionDeclarations" in v:
            return [v]
        # If it's already a list, assume it's correct.
        if isinstance(v, list):
            return v
        # If it's something else, raise an error.
        raise ValueError(
            "Invalid format for 'tools'. Expected a list or a dict with 'functionDeclarations'."
        )


class SafetyRating(LoggingBaseModel):
    category: str
    probability: str
    blocked: Optional[bool] = None
    probabilityScore: Optional[float] = None
    severity: Optional[str] = None
    severityScore: Optional[float] = None


class GroundingMetadata(LoggingBaseModel):
    webSearchQueries: Optional[List[str]] = None
    searchEntryPoint: Optional[Dict[str, Any]] = None
    retrievalMetadata: Optional[Dict[str, Any]] = None


class GeminiCandidate(LoggingBaseModel):
    content: GeminiContent
    finishReason: Optional[str] = None
    index: int = 0
    groundingMetadata: Optional[GroundingMetadata] = None
    safetyRatings: Optional[List[SafetyRating]] = None
    avgLogprobs: Optional[float] = None


class GeminiUsageMetadata(LoggingBaseModel):
    promptTokenCount: Optional[int] = None
    candidatesTokenCount: Optional[int] = None
    totalTokenCount: Optional[int] = None
    trafficType: Optional[str] = None
    promptTokensDetails: Optional[List[Dict[str, Any]]] = None
    candidatesTokensDetails: Optional[List[Dict[str, Any]]] = None
    thoughtsTokenCount: Optional[int] = None


class PromptFeedback(LoggingBaseModel):
    blockReason: Optional[str] = None
    safetyRatings: Optional[List[SafetyRating]] = None


class GeminiResponse(LoggingBaseModel):
    candidates: List[GeminiCandidate]
    promptFeedback: Optional[PromptFeedback] = None
    usageMetadata: Optional[GeminiUsageMetadata] = None
    modelVersion: Optional[str] = None
    responseId: Optional[str] = None
    createTime: Optional[str] = None


class CountTokensRequest(LoggingBaseModel):
    """Request model for the countTokens endpoint."""

    contents: List[GeminiContent]
    systemInstruction: Optional[GeminiSystemInstruction] = None
    tools: Optional[List[Dict[str, Any]]] = None

    class Config:
        extra = "allow"


class EmbedContentRequest(LoggingBaseModel):
    """Request model for the embedContent endpoint."""

    content: GeminiContent
    taskType: Optional[str] = None
    outputDimensionality: Optional[int] = None

    class Config:
        extra = "allow"


class BatchEmbedRequestItem(LoggingBaseModel):
    """An item within a batch embedding request."""

    model: str
    content: GeminiContent
    taskType: Optional[str] = None
    title: Optional[str] = None
    outputDimensionality: Optional[int] = None

    class Config:
        extra = "allow"


class BatchEmbedContentsRequest(LoggingBaseModel):
    """Request model for the batchEmbedContents endpoint."""

    requests: List[BatchEmbedRequestItem]


class CountTokensResponse(LoggingBaseModel):
    totalTokens: int


class ContentEmbedding(LoggingBaseModel):
    values: List[float]


class EmbedContentResponse(LoggingBaseModel):
    embedding: ContentEmbedding


class BatchEmbedContentResponse(LoggingBaseModel):
    embeddings: List[ContentEmbedding]

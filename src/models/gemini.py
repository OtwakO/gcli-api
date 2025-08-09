from typing import Any, Dict, List, Optional

from pydantic import BaseModel


# Gemini Models
class GeminiFunctionCall(BaseModel):
    name: str
    args: Dict[str, Any]


class GeminiFunctionResponse(BaseModel):
    name: str
    response: Dict[str, Any]


class GeminiFileData(BaseModel):
    mimeType: str
    fileUri: str


class GeminiInlineData(BaseModel):
    mimeType: str
    data: str


class GeminiExecutableCode(BaseModel):
    language: str
    code: str


class GeminiCodeExecutionResult(BaseModel):
    outcome: str
    output: str


class GeminiPart(BaseModel):
    text: Optional[str] = None
    inlineData: Optional[GeminiInlineData] = None
    functionCall: Optional[GeminiFunctionCall] = None
    functionResponse: Optional[GeminiFunctionResponse] = None
    fileData: Optional[GeminiFileData] = None
    executableCode: Optional[GeminiExecutableCode] = None
    codeExecutionResult: Optional[GeminiCodeExecutionResult] = None
    thought: Optional[bool] = None
    thoughtSignature: Optional[str] = None


class GeminiContent(BaseModel):
    role: str
    parts: List[GeminiPart] = []


class GeminiSystemInstruction(BaseModel):
    parts: List[GeminiPart]


class GeminiFunctionCallingConfig(BaseModel):
    mode: Optional[str] = None  # "AUTO", "ANY", "NONE"
    allowedFunctionNames: Optional[List[str]] = None


class GeminiToolConfig(BaseModel):
    functionCallingConfig: Optional[GeminiFunctionCallingConfig] = None


class GeminiRequest(BaseModel):
    contents: List[GeminiContent]
    systemInstruction: Optional[GeminiSystemInstruction] = None
    tools: Optional[List[Dict[str, Any]]] = None
    toolConfig: Optional[GeminiToolConfig] = None
    safetySettings: Optional[List[Dict[str, Any]]] = None
    generationConfig: Optional[Dict[str, Any]] = None
    cachedContent: Optional[str] = None


class GroundingMetadata(BaseModel):
    webSearchQueries: Optional[List[str]] = None
    searchEntryPoint: Optional[Dict[str, Any]] = None
    retrievalMetadata: Optional[Dict[str, Any]] = None


class GeminiCandidate(BaseModel):
    content: GeminiContent
    finishReason: Optional[str] = None
    index: int = 0
    groundingMetadata: Optional[GroundingMetadata] = None
    safetyRatings: Optional[List[Dict[str, Any]]] = None
    avgLogprobs: Optional[float] = None


class GeminiUsageMetadata(BaseModel):
    promptTokenCount: Optional[int] = None
    candidatesTokenCount: Optional[int] = None
    totalTokenCount: Optional[int] = None
    trafficType: Optional[str] = None
    promptTokensDetails: Optional[List[Dict[str, Any]]] = None
    candidatesTokensDetails: Optional[List[Dict[str, Any]]] = None
    thoughtsTokenCount: Optional[int] = None


class GeminiResponse(BaseModel):
    candidates: List[GeminiCandidate]
    usageMetadata: Optional[GeminiUsageMetadata] = None


class CountTokensResponse(BaseModel):
    totalTokens: int


class ContentEmbedding(BaseModel):
    values: List[float]


class EmbedContentResponse(BaseModel):
    embedding: ContentEmbedding


class BatchEmbedContentResponse(BaseModel):
    embeddings: List[ContentEmbedding]

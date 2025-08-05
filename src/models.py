from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any

# Tool Calling Models
class FunctionCall(BaseModel):
    name: str
    arguments: str

class ToolCall(BaseModel):
    id: str
    type: str = "function"
    function: FunctionCall

# OpenAI Models
class OpenAIChatMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]], None] = None
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None

class OpenAIChatCompletionRequest(BaseModel):
    model: str
    messages: List[OpenAIChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    n: Optional[int] = None
    seed: Optional[int] = None
    response_format: Optional[Dict[str, Any]] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    
    class Config:
        extra = "allow"

class OpenAIChatCompletionChoice(BaseModel):
    index: int
    message: OpenAIChatMessage
    finish_reason: Optional[str] = None

class OpenAIChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatCompletionChoice]

class OpenAIDelta(BaseModel):
    content: Optional[str] = None
    tool_calls: Optional[List[ToolCall]] = None
    role: Optional[str] = None

class OpenAIChatCompletionStreamChoice(BaseModel):
    index: int
    delta: OpenAIDelta
    finish_reason: Optional[str] = None

class OpenAIChatCompletionStreamResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[OpenAIChatCompletionStreamChoice]

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

class GeminiContent(BaseModel):
    role: str
    parts: List[GeminiPart]

class GeminiSystemInstruction(BaseModel):
    parts: List[GeminiPart]

class GeminiRequest(BaseModel):
    contents: List[GeminiContent]
    systemInstruction: Optional[GeminiSystemInstruction] = None
    tools: Optional[List[Dict[str, Any]]] = None
    toolConfig: Optional[Dict[str, Any]] = None
    safetySettings: Optional[List[Dict[str, Any]]] = None
    generationConfig: Optional[Dict[str, Any]] = None
    cachedContent: Optional[str] = None

class GeminiCandidate(BaseModel):
    content: GeminiContent
    finish_reason: Optional[str] = None
    index: int = 0

class GeminiResponse(BaseModel):
    candidates: List[GeminiCandidate]

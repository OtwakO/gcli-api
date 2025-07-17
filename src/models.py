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

class GeminiPart(BaseModel):
    text: Optional[str] = None
    functionCall: Optional[GeminiFunctionCall] = None

class GeminiContent(BaseModel):
    role: str
    parts: List[GeminiPart]

class GeminiRequest(BaseModel):
    contents: List[GeminiContent]
    tools: Optional[List[Dict[str, Any]]] = None

class GeminiCandidate(BaseModel):
    content: GeminiContent
    finish_reason: Optional[str] = None
    index: int

class GeminiResponse(BaseModel):
    candidates: List[GeminiCandidate]

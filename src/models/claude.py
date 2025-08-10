from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel


# Claude Models
class ClaudeMessage(BaseModel):
    role: str
    content: Union[str, List[Dict[str, Any]]]


class ClaudeMessagesRequest(BaseModel):
    id: Optional[str] = None
    model: str
    messages: List[ClaudeMessage]
    system: Optional[Union[str, List[Dict[str, Any]]]] = None
    max_tokens: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None


# --- Claude Response Models ---


class ClaudeUsage(BaseModel):
    input_tokens: int
    output_tokens: int


class ClaudeContentBlock(BaseModel):
    type: str
    text: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    input: Optional[Dict[str, Any]] = None


class ClaudeMessageResponse(BaseModel):
    id: str
    type: str = "message"
    role: str = "assistant"
    content: List[ClaudeContentBlock]
    model: str
    stop_reason: Optional[str] = None
    stop_sequence: Optional[str] = None
    usage: ClaudeUsage


# --- Claude Streaming Models ---


class ClaudeMessageStartEvent(BaseModel):
    type: Literal["message_start"] = "message_start"
    message: ClaudeMessageResponse


class ClaudeContentBlockStart(BaseModel):
    type: Literal["content_block_start"] = "content_block_start"
    index: int
    content_block: ClaudeContentBlock


class ClaudeTextDelta(BaseModel):
    type: Literal["text_delta"] = "text_delta"
    text: str


class ClaudeInputJsonDelta(BaseModel):
    type: Literal["input_json_delta"] = "input_json_delta"
    partial_json: str


class ClaudeContentBlockDelta(BaseModel):
    type: Literal["content_block_delta"] = "content_block_delta"
    index: int
    delta: Union[ClaudeTextDelta, ClaudeInputJsonDelta]


class ClaudeContentBlockStop(BaseModel):
    type: Literal["content_block_stop"] = "content_block_stop"
    index: int


class ClaudeMessageDelta(BaseModel):
    type: Literal["message_delta"] = "message_delta"
    delta: Dict[str, Any]
    usage: Dict[str, int]


class ClaudeMessageStop(BaseModel):
    type: Literal["message_stop"] = "message_stop"

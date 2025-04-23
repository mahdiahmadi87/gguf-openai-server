import time
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict, Literal
from .common import UsageInfo

# --- Request ---
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"] # Maybe function/tool later
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1 # Number of chat completion choices
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None # Often required
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

    # llama-cpp-python specific parameters mapping
    top_k: Optional[int] = 40
    repeat_penalty: Optional[float] = 1.1

    class Config:
        extra = 'allow' # Allow fields not explicitly defined

# --- Response ---
class ChatCompletionMessage(BaseModel):
    role: Optional[Literal["assistant", "tool"]] = "assistant" # Role of the message creator
    content: Optional[str] = None # Message content (can be null for tool calls)
    # tool_calls: Optional[List[Any]] = None # For function/tool calling later

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None # e.g., "stop", "length", "tool_calls"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}") # Generate a pseudo-ID
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str # Model ID used
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None # Can be added if needed

# --- Streaming Response ---
class ChatCompletionStreamDelta(BaseModel):
    role: Optional[Literal["assistant", "tool"]] = None
    content: Optional[str] = None
    # tool_calls: Optional[List[Any]] = None # For function/tool calling later

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatCompletionStreamDelta # The changes in this chunk
    finish_reason: Optional[str] = None # Usually null until the last chunk

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion.chunk" # Note the ".chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str # Model ID
    choices: List[ChatCompletionStreamChoice]
    usage: Optional[UsageInfo] = None # Typically null or only in last chunk
    system_fingerprint: Optional[str] = None
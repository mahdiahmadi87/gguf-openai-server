# llm_server/schemas/chat.py
import time
from pydantic import BaseModel, Field, HttpUrl, field_validator, ValidationError
from typing import Optional, List, Union, Dict, Literal, Any

from .common import UsageInfo

# --- Content Part Schemas ---
class TextContentPart(BaseModel):
    type: Literal["text"] = "text"
    text: str

class ImageURL(BaseModel):
    url: str # Can be data: URI or potentially http/https URL (though we'll only support data URI initially)
    detail: Optional[Literal["low", "high", "auto"]] = "auto"

    @field_validator('url')
    @classmethod
    def validate_url_format(cls, v):
        if not (v.startswith("data:image/") or v.startswith("http://") or v.startswith("https://")):
             raise ValueError("URL must start with 'data:image/', 'http://', or 'https://'")
        if v.startswith("data:image/") and ";base64," not in v:
             raise ValueError("Data URL format invalid. Expected 'data:image/...;base64,...'")
        return v


class ImageContentPart(BaseModel):
    type: Literal["image_url"] = "image_url"
    image_url: ImageURL


# --- Chat Message Schema ---
class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] # Added tool role
    # Content can be a simple string OR a list of content parts (for multimodal)
    content: Union[str, List[Union[TextContentPart, ImageContentPart]]]
    name: Optional[str] = None

    # Optional: Add a validator to ensure list content isn't empty if it's a list
    @field_validator('content')
    @classmethod
    def check_content_list_not_empty(cls, v):
        if isinstance(v, list) and not v:
             raise ValueError("If content is a list, it must not be empty.")
        return v


# --- Chat Completion Request ---
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage] # Uses the updated ChatMessage schema
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # --- llama-cpp specific ---
    top_k: Optional[int] = 40
    repeat_penalty: Optional[float] = 1.1

    class Config:
        extra = 'allow'


# --- Response Schemas (ChatCompletionResponse, StreamResponse, etc.) ---
# These generally don't need changes unless the *output* format changes significantly
# Keep ChatCompletionMessage content as Optional[str] for now, as models typically respond with text.
class ChatCompletionMessage(BaseModel):
    role: Optional[Literal["assistant", "tool"]] = "assistant"
    content: Optional[str] = None
    # tool_calls: Optional[List[Any]] = None

# ... (Rest of the response/stream schemas remain the same) ...
class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None

class ChatCompletionStreamDelta(BaseModel):
    role: Optional[Literal["assistant", "tool"]] = None
    content: Optional[str] = None
    # tool_calls: Optional[List[Any]] = None

class ChatCompletionStreamChoice(BaseModel):
    index: int
    delta: ChatCompletionStreamDelta
    finish_reason: Optional[str] = None

class ChatCompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionStreamChoice]
    usage: Optional[UsageInfo] = None
    system_fingerprint: Optional[str] = None
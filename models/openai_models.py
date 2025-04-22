# models/openai_models.py
# Pydantic models mirroring OpenAI API structures

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union, Literal, Any
import time
import uuid

# --- Base Models ---

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: Optional[int] = 0
    total_tokens: int = 0

# --- Chat Completions Models ---

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: Optional[str] = None
    # Add tool_calls and tool_call_id if needed for tool usage support

class ChatCompletionRequest(BaseModel):
    model: str # Although we load one model, client might specify compatible names
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1 # Number of completions to generate
    max_tokens: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None # Identifier for the end-user

    # llama-cpp-python specific parameters can be added if needed
    # e.g., top_k: Optional[int] = 40
    #       repeat_penalty: Optional[float] = 1.1

class ChatCompletionChoiceDelta(BaseModel):
    role: Optional[Literal["system", "user", "assistant", "tool"]] = None
    content: Optional[str] = None
    # tool_calls: Optional[...] = None # For tool usage streaming

class ChatCompletionStreamResponseDelta(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[Literal["index", "delta", "finish_reason"], Any]] = []
    # Example choice structure: [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}]
    # Example finish choice: [{"index": 0, "delta": {}, "finish_reason": "stop"}]

class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo

# --- (Legacy) Completions Models ---
# Optional: Implement if compatibility with the older /v1/completions is needed

class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    logprobs: Optional[int] = None
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None

class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    logprobs: Optional[Any] = None # Can be complex structure
    finish_reason: Optional[Literal["stop", "length"]] = None

class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo

class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex}")
    object: Literal["text_completion"] = "text_completion" # Note: OpenAI uses 'text_completion' for the object type in streaming too
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Dict[Literal["index", "text", "logprobs", "finish_reason"], Any]] = []
    # Example choice structure: [{"index": 0, "text": " response chunk", "logprobs": None, "finish_reason": None}]
    # Example finish choice: [{"index": 0, "text": "", "logprobs": None, "finish_reason": "stop"}]
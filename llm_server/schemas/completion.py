import time
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Dict, Any
from .common import UsageInfo, ChoiceLogProbs

# --- Request ---
class CompletionRequest(BaseModel):
    model: str
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1 # Number of choices to generate
    stream: Optional[bool] = False
    logprobs: Optional[int] = None # Number of logprobs to return
    echo: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    best_of: Optional[int] = 1 # Not always supported well locally
    logit_bias: Optional[Dict[str, float]] = None # Map of token IDs to bias
    user: Optional[str] = None # Identifier for end-user monitoring

    # llama-cpp-python specific parameters mapping
    top_k: Optional[int] = 40 # Often used instead of top_p by llama.cpp
    repeat_penalty: Optional[float] = 1.1 # Alias for frequency_penalty? Check llama-cpp-python docs

    class Config:
        extra = 'allow' # Allow fields not explicitly defined, useful for model-specific params


# --- Response ---
class CompletionChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[ChoiceLogProbs] = None
    finish_reason: Optional[str] = None # e.g., "stop", "length"

class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{int(time.time())}") # Generate a pseudo-ID
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str # The model ID used for the completion
    choices: List[CompletionChoice]
    usage: Optional[UsageInfo] = None

# --- Streaming Response ---
class CompletionStreamChoice(BaseModel):
    index: int
    text: str
    logprobs: Optional[ChoiceLogProbs] = None # Usually null in stream chunks until the end?
    finish_reason: Optional[str] = None

class CompletionStreamResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{int(time.time())}")
    object: str = "text_completion.chunk" # Note the ".chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str # Model ID
    choices: List[CompletionStreamChoice]
    # Usage typically not present in chunks, maybe in the final one?
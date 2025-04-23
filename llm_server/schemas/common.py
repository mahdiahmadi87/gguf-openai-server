from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Union

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: Optional[int] = 0
    total_tokens: int = 0

class LogProbs(BaseModel):
    # Simplified - adjust if full logprobs needed
    token: str
    logprob: float

class TopLogProbs(BaseModel):
    # Simplified
    token: str
    logprob: float
    bytes: Optional[List[int]] = None

class ChoiceLogProbs(BaseModel):
     tokens: Optional[List[str]] = None
     token_logprobs: Optional[List[Optional[float]]] = None
     top_logprobs: Optional[List[Optional[Dict[str, float]]]] = None # Or List[Optional[TopLogProbs]]
     text_offset: Optional[List[int]] = None
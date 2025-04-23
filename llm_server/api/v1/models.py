from fastapi import APIRouter, Depends, HTTPException, status
from typing import List
from pydantic import BaseModel, Field
import time

from llm_server.core.security import validate_api_key
from config.settings import get_model_config, get_available_model_ids

router = APIRouter(tags=["Models"])

class Model(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "local" # Or derive from config if needed

class ModelList(BaseModel):
    object: str = "list"
    data: List[Model]

@router.get(
    "/models",
    response_model=ModelList,
    dependencies=[Depends(validate_api_key)] # Protect this endpoint
)
async def list_models():
    """Lists the currently available models."""
    available_model_ids = get_available_model_ids()
    model_cards = [
        Model(id=model_id) for model_id in available_model_ids
    ]
    if not model_cards:
         # Optional: Return 404 if no models are configured? Or empty list?
         # Let's return empty list for now, consistent with OpenAI if no models of a type exist.
         pass

    return ModelList(data=model_cards)
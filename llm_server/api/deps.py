# llm_server/api/deps.py
from fastapi import Depends, HTTPException, status, Request
from llm_server.core.security import validate_api_key
from llm_server.core.llm_manager import get_llama_instance, get_model_config
from config.settings import get_available_model_ids
from llama_cpp import Llama

# Keep the simple API key validation dependency
async def get_authenticated_user(
    _: None = Depends(validate_api_key) # Renamed for clarity, just runs validation
):
    """Dependency that only performs API key validation."""
    pass # Validation happens in validate_api_key


# The model instance retrieval will now happen *inside* the endpoint,
# as we need the request body first. We remove the combined dependencies.

# Helper function (can stay here or move to utils) to validate model ID
def validate_model_id(model_id: str) -> str:
    """Checks if the requested model_id is configured and available."""
    if model_id not in get_available_model_ids():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model '{model_id}' not found or not configured."
        )
    # You could add the path check here again if desired, but it's checked at startup
    config = get_model_config(model_id)
    if not config or not config.model_path:
         raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, # Or 500
            detail=f"Model '{model_id}' configuration is invalid or path is missing."
        )
    return model_id

# Helper function to get model instance (can stay here or move to utils)
def get_model_instance_from_id(model_id: str) -> Llama:
    """Gets the loaded Llama instance for the validated model_id."""
    try:
        # Handles loading and caching
        return get_llama_instance(model_id)
    except ValueError as e: # Config not found
         raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except RuntimeError as e: # Loading error
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    except Exception as e:
         print(f"Unexpected error getting model instance for '{model_id}': {e}")
         raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while accessing model.")
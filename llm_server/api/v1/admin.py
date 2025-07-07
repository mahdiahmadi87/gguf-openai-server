# llm_server/api/v1/admin.py
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from llm_server.core.api_key_manager import add_api_key, remove_api_key
from llm_server.core.security import validate_admin_api_key

# Define the router, protected by the admin key dependency at the router level
router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(validate_admin_api_key)]
)

# --- Pydantic Schemas for Request Bodies ---
class APIKeyBody(BaseModel):
    api_key: str = Field(..., min_length=10, description="The API key to add or remove.")

class APIKeyResponse(BaseModel):
    message: str
    api_key: str

# --- Endpoints ---
@router.post(
    "/keys",
    response_model=APIKeyResponse,
    summary="Add a new API Key",
    status_code=status.HTTP_201_CREATED,
)
async def create_new_api_key(body: APIKeyBody):
    """
    Adds a new API key to the list of allowed keys.
    The key will be persisted across server restarts.
    """
    success = add_api_key(body.api_key)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="API key already exists.",
        )
    return APIKeyResponse(message="API key added successfully.", api_key=body.api_key)


@router.delete(
    "/keys",
    response_model=APIKeyResponse,
    summary="Revoke an existing API Key",
    status_code=status.HTTP_200_OK,
)
async def revoke_api_key(body: APIKeyBody):
    """
    Removes an API key from the list of allowed keys, revoking its access.
    """
    success = remove_api_key(body.api_key)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found.",
        )
    return APIKeyResponse(message="API key revoked successfully.", api_key=body.api_key)
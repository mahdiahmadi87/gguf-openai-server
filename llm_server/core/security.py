# llm_server/core/security.py
from fastapi import HTTPException, status, Security
from fastapi.security import APIKeyHeader

from config.settings import settings
from llm_server.core.api_key_manager import is_key_valid # <-- Import from new manager

# --- User API Key Validation ---
USER_API_KEY_NAME = "Authorization"
user_api_key_header = APIKeyHeader(name=USER_API_KEY_NAME, auto_error=True)

async def validate_api_key(api_key_header: str = Security(user_api_key_header)):
    """Dependency to validate the API key for regular user endpoints."""
    if not api_key_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme. Use 'Bearer'.",
        )
    token = api_key_header.split("Bearer ")[1]

    # Use the new key manager for validation
    if not is_key_valid(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

# --- NEW: Admin API Key Validation ---
ADMIN_API_KEY_NAME = "X-Admin-API-Key"
admin_api_key_header = APIKeyHeader(name=ADMIN_API_KEY_NAME, auto_error=True)

async def validate_admin_api_key(api_key_header: str = Security(admin_api_key_header)):
    """Dependency to validate the ADMIN_API_KEY for protected admin endpoints."""
    if not settings.ADMIN_API_KEY:
        # This case should ideally not be reached if settings load correctly
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Admin API Key is not configured on the server."
        )
    if api_key_header != settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid Admin API Key.",
        )
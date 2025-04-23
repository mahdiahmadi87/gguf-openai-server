from typing import Optional
from fastapi import HTTPException, status, Security
from fastapi.security import APIKeyHeader
from config.settings import settings

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def validate_api_key(api_key_header: str = Security(api_key_header)):
    """Dependency to validate the API key provided in the Authorization header."""
    if not api_key_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization scheme. Use 'Bearer'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = api_key_header.split("Bearer ")[1]

    if not settings.ALLOWED_API_KEYS:
        # If no keys are configured, potentially allow all or deny all.
        # Denying all is safer if keys are expected.
        print("WARNING: No API keys configured in ALLOWED_API_KEYS. Denying request.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access denied. API keys are not configured on the server.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if token not in settings.ALLOWED_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    # Return the token if needed downstream, otherwise just validating is enough
    # return token
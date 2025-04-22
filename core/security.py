# core/security.py
from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
import logging

from .config import settings # Import the initialized settings

logger = logging.getLogger(__name__)

API_KEY_NAME = "Authorization"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False) # auto_error=False to handle missing header manually

async def get_api_key(api_key_header: str = Security(api_key_header)):
    """
    Dependency to validate the API key provided in the Authorization header.
    Expected format: "Authorization: Bearer YOUR_API_KEY"
    """
    # If no keys are configured, skip authentication
    if not settings.API_KEYS:
        logger.debug("API key authentication skipped (no keys configured).")
        return "dummy_key_unauthenticated" # Return a placeholder

    if not api_key_header:
        logger.warning("Missing Authorization header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
        )

    # Extract the key from "Bearer <key>"
    parts = api_key_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        logger.warning(f"Invalid Authorization header format: {api_key_header}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <api_key>'",
        )

    api_key = parts[1]

    if api_key not in settings.API_KEYS:
        logger.warning(f"Invalid API Key received: {api_key[:5]}...") # Log only prefix for security
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )

    logger.debug(f"Valid API Key received: {api_key[:5]}...")
    return api_key # Return the validated key (could be used for logging/auditing)


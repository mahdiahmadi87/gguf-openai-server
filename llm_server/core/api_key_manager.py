# llm_server/core/api_key_manager.py
import json
import os
import logging
from threading import Lock
from typing import Set

from config.settings import settings

logger = logging.getLogger(__name__)

# In-memory set for fast lookups
_api_keys: Set[str] = set()
# Thread lock to prevent race conditions when reading/writing the file
_file_lock = Lock()


def load_api_keys():
    """
    Loads API keys from the JSON file into the in-memory set.
    This should be called once at server startup.
    """
    global _api_keys
    with _file_lock:
        try:
            # Create the file with an empty list if it doesn't exist
            if not os.path.exists(settings.API_KEYS_FILE):
                logger.info(f"API keys file not found. Creating a new one at '{settings.API_KEYS_FILE}'.")
                with open(settings.API_KEYS_FILE, "w") as f:
                    json.dump([], f)
                _api_keys = set()
                return

            # Read existing keys from the file
            with open(settings.API_KEYS_FILE, "r") as f:
                keys_from_file = json.load(f)
                if not isinstance(keys_from_file, list):
                    raise TypeError("API keys file should contain a JSON list of strings.")
                _api_keys = set(keys_from_file)
                logger.info(f"Successfully loaded {len(_api_keys)} API keys from '{settings.API_KEYS_FILE}'.")

        except (json.JSONDecodeError, TypeError) as e:
            logger.error(f"Error reading or parsing API keys file '{settings.API_KEYS_FILE}': {e}. Please ensure it's a valid JSON list.")
            # Decide on fallback behavior: fail hard or run with no keys?
            # Running with no keys is safer if the file is corrupted.
            _api_keys = set()
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading API keys: {e}", exc_info=True)
            _api_keys = set()

def _save_api_keys():
    """
    Saves the current in-memory set of keys back to the JSON file.
    This is an internal function and should always be called within a lock.
    """
    with open(settings.API_KEYS_FILE, "w") as f:
        json.dump(list(_api_keys), f, indent=2)

def is_key_valid(api_key: str) -> bool:
    """Checks if a given API key is in the current set of valid keys."""
    return api_key in _api_keys

def add_api_key(api_key: str) -> bool:
    """
    Adds a new API key to the set and file.
    Returns True if the key was added, False if it already existed.
    """
    with _file_lock:
        if api_key in _api_keys:
            logger.warning(f"Attempted to add an API key that already exists: '{api_key[:8]}...'")
            return False
        _api_keys.add(api_key)
        _save_api_keys()
        logger.info(f"Successfully added new API key: '{api_key[:8]}...'")
        return True

def remove_api_key(api_key: str) -> bool:
    """
    Removes an API key from the set and file.
    Returns True if the key was removed, False if it was not found.
    """
    with _file_lock:
        if api_key not in _api_keys:
            logger.warning(f"Attempted to remove an API key that was not found: '{api_key[:8]}...'")
            return False
        _api_keys.remove(api_key)
        _save_api_keys()
        logger.info(f"Successfully removed API key: '{api_key[:8]}...'")
        return True
# core/config.py
import os
from dotenv import load_dotenv
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Load environment variables from a .env file if it exists
load_dotenv()

class Settings:
    # --- Model Configuration ---
    # Option 1: Direct path
    MODEL_PATH: Optional[str] = os.getenv("MODEL_PATH")

    # Option 2: Hugging Face repo (requires HF_MODEL_FILE to be set)
    # HF_MODEL_REPO_ID: Optional[str] = os.getenv("HF_MODEL_REPO_ID")
    # HF_MODEL_FILE: Optional[str] = os.getenv("HF_MODEL_FILE") # e.g., "mistral-7b-instruct-v0.1.Q4_K_M.gguf"

    # --- llama-cpp-python Parameters ---
    # These are examples, adjust based on your needs and model capabilities
    N_GPU_LAYERS: int = int(os.getenv("N_GPU_LAYERS", 0)) # -1 for all layers, 0 for CPU only
    N_CTX: int = int(os.getenv("N_CTX", 4096)) # Context window size
    N_BATCH: int = int(os.getenv("N_BATCH", 512)) # Batch size for prompt processing
    LLAMA_VERBOSE: bool = os.getenv("LLAMA_VERBOSE", "False").lower() == "true"

    # --- API Server Configuration ---
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    API_KEYS_STR: Optional[str] = os.getenv("API_KEYS") # Comma-separated list of valid keys
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # --- Derived/Processed Configuration ---
    API_KEYS: List[str] = []
    if API_KEYS_STR:
        API_KEYS = [key.strip() for key in API_KEYS_STR.split(",")]
    else:
        logger.warning("API_KEYS environment variable not set. Authentication will be disabled.")
        # Consider adding a default dummy key for local testing if desired,
        # but be cautious about security implications.
        # API_KEYS = ["your-default-test-key"]

    # --- Model Name ---
    # This is the identifier the client will use in requests.
    # It should ideally match the model loaded, but can be customized.
    MODEL_NAME: str = os.getenv("MODEL_NAME", "local-llm")

    def __init__(self):
        # --- Validation ---
        if not self.MODEL_PATH:
             # Add logic here if using Hugging Face download instead
             # if not self.HF_MODEL_REPO_ID or not self.HF_MODEL_FILE:
             raise ValueError("MODEL_PATH environment variable must be set.")
        if self.MODEL_PATH and not os.path.exists(self.MODEL_PATH):
             raise FileNotFoundError(f"Model file not found at path: {self.MODEL_PATH}")

        if not self.API_KEYS:
            logger.warning("Running without API key authentication. The API is open.")


# Instantiate settings
try:
    settings = Settings()
except (ValueError, FileNotFoundError) as e:
    logger.error(f"Configuration Error: {e}")
    # Exit or raise depending on desired behavior if config is invalid
    import sys
    sys.exit(1)


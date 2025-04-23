import os
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file variables
load_dotenv()
        
class ModelInfo(BaseSettings):
    """Configuration for a single GGUF model."""
    model_id: str  # User-facing model name (e.g., "mistral-7b-instruct-v0.1")
    model_path: str # Absolute path to the .gguf file
    # llama-cpp-python parameters (add more as needed)
    n_gpu_layers: int = -1 # -1 means attempt to offload all layers to GPU
    n_ctx: int = 4096     # Default context size
    # Add other Llama() constructor args here if needed:
    # e.g., n_batch, logits_all, etc.


class Settings(BaseSettings):
    """Main application settings."""
    APP_NAME: str = "Local OpenAI-Compatible LLM Server"
    BASE_PATH: str = "/v1" # API prefix

    # --- Model Configuration ---
    # Define models here directly or load from another source (e.g., JSON file)
    # Example: Load from environment variable MODELS_CONFIG='[{"model_id": "mistral-7b", "model_path": "/path/to/models/mistral-7b.Q4_K_M.gguf"}]'
    # Or define directly:
    MODELS: List[ModelInfo] = [
        # Add your models here
        # Example:
        # ModelInfo(model_id="mistral-7b-instruct", model_path="/path/to/your/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", n_gpu_layers=-1),
        # ModelInfo(model_id="llama-2-7b-chat", model_path="/path/to/your/models/llama-2-7b-chat.Q4_K_M.gguf", n_gpu_layers=-1),
        ModelInfo(model_id="gemma-3-4b", model_path="/home/mahdi/Documents/LLM/gguf-openai-server/models/gemma-3-4b-it-q4_0.gguf", n_gpu_layers=-1),
        ModelInfo(model_id="aya-8b", model_path="/home/mahdi/Documents/LLM/gguf-openai-server/models/aya-expanse-8b-q4_0.gguf", n_gpu_layers=-1),
    ]

    # --- Security ---
    # Load allowed API keys from environment variable (comma-separated)
    # Example: export ALLOWED_API_KEYS="sk-local-key1,sk-local-key2"
    ALLOWED_API_KEYS_STR: Optional[str] = os.getenv("ALLOWED_API_KEYS", None)

    @property
    def ALLOWED_API_KEYS(self) -> List[str]:
        if self.ALLOWED_API_KEYS_STR:
            return [key.strip() for key in self.ALLOWED_API_KEYS_STR.split(',')]
        return []

    # --- Server Configuration ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # --- Logging ---
    LOG_LEVEL: str = "INFO"

    # --- llama-cpp-python specific - can be overridden per model ---
    DEFAULT_N_GPU_LAYERS: int = -1
    DEFAULT_N_CTX: int = 4096
    VERBOSE_LLAMA: bool = False # Set llama.cpp verbosity

    # --- Rate Limiting (Optional) ---
    # RATE_LIMIT_ENABLED: bool = False
    # RATE_LIMIT_DEFAULT: str = "10/minute" # Example: 10 requests per minute

    class Config:
        # If loading complex structures like MODELS from .env, you might need custom parsing
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore' # Ignore extra fields from environment

# Instantiate settings
settings = Settings()

# --- Model Path Validation ---
# Ensure configured model paths exist
_MODEL_MAP: Dict[str, ModelInfo] = {}
for model_info in settings.MODELS:
    if not os.path.exists(model_info.model_path):
        print(f"WARNING: Model path not found for '{model_info.model_id}': {model_info.model_path}")
        # Decide whether to raise an error or just warn
        # raise FileNotFoundError(f"Model path not found for '{model_info.model_id}': {model_info.model_path}")
    else:
        _MODEL_MAP[model_info.model_id] = model_info

# Make the validated map accessible easily
def get_model_config(model_id: str) -> Optional[ModelInfo]:
    return _MODEL_MAP.get(model_id)

def get_available_model_ids() -> List[str]:
    return list(_MODEL_MAP.keys())

# --- Print loaded settings for verification ---
print("--- Server Configuration ---")
print(f"Host: {settings.HOST}")
print(f"Port: {settings.PORT}")
print(f"Allowed API Keys: {'Loaded' if settings.ALLOWED_API_KEYS else 'None'}")
print("Configured Models:")
if _MODEL_MAP:
    for model_id, info in _MODEL_MAP.items():
        print(f"  - ID: {model_id}, Path: {info.model_path}, GPU Layers: {info.n_gpu_layers}, Context: {info.n_ctx}")
else:
    print("  No valid models configured or found.")
print("--------------------------")

# Check if any models are actually configured and valid
if not _MODEL_MAP:
    print("ERROR: No valid models configured. Please check your .env file or settings.py and ensure model paths are correct.")
    # Exit if no models are available? Or let the server start but fail requests?
    # For now, just print the error. The server will start but endpoints needing models will fail.
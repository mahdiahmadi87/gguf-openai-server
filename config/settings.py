# config/settings.py
import os
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class ModelInfo(BaseSettings):
    """Configuration for a single GGUF model."""
    model_id: str
    model_path: str
    n_gpu_layers: int = -1
    n_ctx: int = 4096
    is_multimodal: bool = False
    # Add other Llama() constructor args here if needed

class Settings(BaseSettings):
    """Main application settings."""
    APP_NAME: str = "Local OpenAI-Compatible LLM Server"
    BASE_PATH: str = "/v1"

    MODELS: List[ModelInfo] = [
        # Example - Update with your actual models
        # ModelInfo(model_id="gemma-3-4b", model_path="/home/mahdi/Documents/LLM/gguf-openai-server/models/gemma-3-4b-it-q4_0.gguf", n_gpu_layers=-1, is_multimodal=True), # Assuming Gemma 3 is multimodal
    ]
    # --- Security ---
    ALLOWED_API_KEYS_STR: Optional[str] = os.getenv("ALLOWED_API_KEYS", None)
    @property
    def ALLOWED_API_KEYS(self) -> List[str]:
        if self.ALLOWED_API_KEYS_STR:
            return [key.strip() for key in self.ALLOWED_API_KEYS_STR.split(',')]
        return []

    # --- Server Configuration ---
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    VERBOSE_LLAMA: bool = False

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'

# --- Instantiate settings ---  <-*** ADD THIS LINE ***
settings = Settings()
# ----------------------------

# --- Model Path Validation & Map Creation ---
_MODEL_MAP: Dict[str, ModelInfo] = {}
# Now 'settings.MODELS' can be accessed safely
for model_info in settings.MODELS:
    if not os.path.exists(model_info.model_path):
        print(f"WARNING: Model path not found for '{model_info.model_id}': {model_info.model_path}")
        # Decide whether to raise an error or just warn
    else:
        _MODEL_MAP[model_info.model_id] = model_info

def get_model_config(model_id: str) -> Optional[ModelInfo]:
    return _MODEL_MAP.get(model_id)

def get_available_model_ids() -> List[str]:
    return list(_MODEL_MAP.keys())

# --- Print loaded settings for verification ---
print("--- Server Configuration ---")
print(f"Host: {settings.HOST}") # Access instance here
print(f"Port: {settings.PORT}") # Access instance here
print(f"Allowed API Keys: {'Loaded' if settings.ALLOWED_API_KEYS else 'None'}") # Access instance here
print("Configured Models:")
if _MODEL_MAP:
    for model_id, info in _MODEL_MAP.items():
        # Access attributes from the 'info' object (ModelInfo instance)
        print(f"  - ID: {model_id}, Path: {info.model_path}, GPU Layers: {info.n_gpu_layers}, Context: {info.n_ctx}, Multimodal: {info.is_multimodal}")
else:
    print("  No valid models configured or found.")
print("--------------------------")

if not _MODEL_MAP:
    print("ERROR: No valid models configured. Please check your .env file or settings.py and ensure model paths are correct.")
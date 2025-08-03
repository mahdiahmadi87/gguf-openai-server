# config/settings.py
import os
import json
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()

class ModelInfo(BaseSettings):
    """Configuration for a single GGUF model."""
    model_id: str
    model_path: str
    n_gpu_layers: int = 10
    n_ctx: int = 65536
    n_threads: int = 4  # Number of CPU threads to use for this model
    thread_offset: int = 0  # Starting CPU core number for this model
    is_multimodal: bool = False
    # Add other Llama() constructor args here if needed

class Settings(BaseSettings):
    """Main application settings."""
    APP_NAME: str = "Local OpenAI-Compatible LLM Server"
    BASE_PATH: str = "/v1"

    # --- Model Configuration ---
    # Attempt to load from MODELS_CONFIG env var first, then fall back to hardcoded.
    MODELS_CONFIG_STR: Optional[str] = os.getenv("MODELS_CONFIG", None)
    MODELS: List[ModelInfo] = []

    def __init__(self, **values: Any):
        super().__init__(**values)
        if self.MODELS_CONFIG_STR:
            try:
                models_data = json.loads(self.MODELS_CONFIG_STR)
                self.MODELS = [ModelInfo(**data) for data in models_data]
            except json.JSONDecodeError:
                print(f"WARNING: Could not parse MODELS_CONFIG JSON: {self.MODELS_CONFIG_STR}")
                self._initialize_default_models()
            except Exception as e:
                print(f"WARNING: Error initializing models from MODELS_CONFIG: {e}")
                self._initialize_default_models()
        else:
            self._initialize_default_models()

    def _initialize_default_models(self):
        # This is where your previous hardcoded models would go as a fallback
        # Or leave it empty if .env is the only way to define models
        self.MODELS = [
            # Example - Update with your actual models if you want a fallback
            # ModelInfo(model_id="gemma-3-4b-default", model_path="/path/to/your/default/gemma-3-4b-it-q4_0.gguf", n_gpu_layers=-1, is_multimodal=True),
        ]


    # --- NEW: Admin & Key Management Configuration ---
    # The master key to protect admin endpoints. Must be set in .env
    ADMIN_API_KEY: str

    # The file where user API keys will be stored persistently.
    API_KEYS_FILE: str = "api_keys.json"
    # --- END NEW ---

    # --- REMOVED ---
    # ALLOWED_API_KEYS_STR and its property are no longer used.
    # --- END REMOVED ---

    # --- Server Configuration ---
    HOST: str = os.getenv("HOST")
    PORT: int = os.getenv("PORT")
    LOG_LEVEL: str = "INFO"
    VERBOSE_LLAMA: bool = False

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        extra = 'ignore'

settings = Settings()
# ----------------------------

# --- Model Path Validation & Map Creation ---
_MODEL_MAP: Dict[str, ModelInfo] = {}
for model_info in settings.MODELS:
    if not os.path.exists(model_info.model_path):
        print(f"WARNING: Model path not found for '{model_info.model_id}': {model_info.model_path}")
    else:
        _MODEL_MAP[model_info.model_id] = model_info

def get_model_config(model_id: str) -> Optional[ModelInfo]:
    return _MODEL_MAP.get(model_id)

def get_available_model_ids() -> List[str]:
    return list(_MODEL_MAP.keys())

# --- Print loaded settings for verification ---
print("--- Server Configuration ---")
print(f"Host: {settings.HOST}")
print(f"Port: {settings.PORT}")
print(f"Admin API Key: {'Loaded' if settings.ADMIN_API_KEY else 'NOT SET! The server will fail to start.'}") # New print
print("Configured Models:")
if _MODEL_MAP:
    for model_id, info in _MODEL_MAP.items():
        print(f"  - ID: {model_id}, Path: {info.model_path}, GPU Layers: {info.n_gpu_layers}, Context: {info.n_ctx}, Multimodal: {info.is_multimodal}")
else:
    print("  No valid models configured or found.")
print("--------------------------")

if not _MODEL_MAP:
    print("ERROR: No valid models configured.")
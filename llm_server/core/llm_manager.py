import threading
from typing import Dict, Optional
from llama_cpp import Llama
from config.settings import settings, ModelInfo, get_model_config

# Thread-safe cache for loaded models
loaded_models: Dict[str, Llama] = {}
model_locks: Dict[str, threading.Lock] = {}
cache_lock = threading.Lock()

def get_llama_instance(model_id: str) -> Llama:
    """
    Loads a Llama model instance based on the model_id.
    Uses caching and locking to handle concurrent requests.
    """
    with cache_lock:
        if model_id in loaded_models:
            return loaded_models[model_id]

        # Get lock specific to this model_id to prevent race conditions during loading
        if model_id not in model_locks:
            model_locks[model_id] = threading.Lock()

    # Acquire the specific model lock *outside* the cache lock
    # to allow other models to be accessed/loaded concurrently.
    with model_locks[model_id]:
        # Double-check if model was loaded by another thread while waiting for the lock
        with cache_lock:
            if model_id in loaded_models:
                return loaded_models[model_id]

        # Load the model
        model_config = get_model_config(model_id)
        if not model_config:
            raise ValueError(f"Model configuration not found for model_id: {model_id}")

        print(f"Loading model '{model_id}' from {model_config.model_path}...")
        try:
            llama_instance = Llama(
                model_path=model_config.model_path,
                n_gpu_layers=model_config.n_gpu_layers,
                n_ctx=model_config.n_ctx,
                verbose=settings.VERBOSE_LLAMA,
                # Add other relevant Llama params from model_config if defined
                # e.g., n_batch=model_config.n_batch
            )

            # Store the loaded instance in the cache
            with cache_lock:
                loaded_models[model_id] = llama_instance
            print(f"Model '{model_id}' loaded successfully.")
            return llama_instance

        except Exception as e:
            # Ensure lock is released if loading fails, prevent deadlock
            # (though context manager 'with' should handle this)
            print(f"Error loading model '{model_id}': {e}")
            # Clean up lock if model failed to load? Maybe not needed if we don't cache failures.
            raise RuntimeError(f"Failed to load model '{model_id}'. Error: {e}")


def unload_model(model_id: str):
    """Unloads a model from memory (if loaded)."""
    with cache_lock:
        if model_id in loaded_models:
            # llama-cpp-python doesn't have an explicit unload,
            # rely on garbage collection by removing reference.
            del loaded_models[model_id]
            if model_id in model_locks:
                del model_locks[model_id] # Remove associated lock
            print(f"Model '{model_id}' unloaded (reference removed).")
        else:
            print(f"Model '{model_id}' not found in cache, nothing to unload.")

# Optional: Preload models at startup?
# def preload_models():
#     print("Preloading configured models...")
#     for model_id in get_available_model_ids():
#         try:
#             get_llama_instance(model_id)
#         except Exception as e:
#             print(f"Failed to preload model '{model_id}': {e}")

# if settings.PRELOAD_MODELS: # Add PRELOAD_MODELS to Settings if needed
#      preload_models()
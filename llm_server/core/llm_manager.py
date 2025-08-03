import threading
import os
import multiprocessing
from typing import Dict, Optional, Set
from llama_cpp import Llama
from config.settings import settings, ModelInfo, get_model_config

def set_cpu_affinity(cpu_ids: Set[int]):
    """Set CPU affinity for the current process."""
    try:
        # Get the current process ID
        pid = os.getpid()
        
        # On Linux, we can use os.sched_setaffinity
        if hasattr(os, 'sched_setaffinity'):
            os.sched_setaffinity(pid, cpu_ids)
        else:
            # Fallback to using taskset via subprocess on Unix-like systems
            import subprocess
            cpu_list = ','.join(map(str, cpu_ids))
            subprocess.run(['taskset', '-pc', cpu_list, str(pid)], check=True)
            
    except Exception as e:
        print(f"Warning: Could not set CPU affinity: {e}")
        # Continue execution even if setting affinity fails

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
            # Calculate CPU core IDs for this model instance
            start_core = model_config.thread_offset
            num_threads = model_config.n_threads
            cpu_cores = set(range(start_core, start_core + num_threads))
            
            # Set CPU affinity for the current thread/process
            set_cpu_affinity(cpu_cores)
            
            # Create the Llama instance with the specified number of threads
            llama_instance = Llama(
                model_path=model_config.model_path,
                n_gpu_layers=model_config.n_gpu_layers,
                n_ctx=model_config.n_ctx,
                verbose=settings.VERBOSE_LLAMA,
                n_threads=model_config.n_threads,  # Set number of threads explicitly
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
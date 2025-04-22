# core/llm_backend.py
import logging
from llama_cpp import Llama
from typing import List, Optional, Dict, Union, Iterator, Any
import time

from .config import settings
from models.openai_models import ChatMessage # Use the defined Pydantic models

logger = logging.getLogger(__name__)

class LLMBackend:
    """
    Wrapper class for the llama-cpp-python model interaction.
    Handles loading the model and generating completions/chat responses.
    """
    _instance = None
    llm = None

    def __new__(cls, *args, **kwargs):
        # Singleton pattern to ensure only one model instance is loaded
        if not cls._instance:
            cls._instance = super(LLMBackend, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._load_model()
        self._initialized = True

    def _load_model(self):
        """Loads the GGUF model using llama-cpp-python."""
        logger.info(f"Loading model from path: {settings.MODEL_PATH}")
        logger.info(f"Using n_gpu_layers: {settings.N_GPU_LAYERS}, n_ctx: {settings.N_CTX}, n_batch: {settings.N_BATCH}")
        try:
            self.llm = Llama(
                model_path=settings.MODEL_PATH,
                n_gpu_layers=settings.N_GPU_LAYERS,
                n_ctx=settings.N_CTX,
                n_batch=settings.N_BATCH,
                verbose=settings.LLAMA_VERBOSE,
                # Add other llama.cpp parameters as needed from settings
                # e.g., seed=42, logits_all=True, etc.
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load LLM model from {settings.MODEL_PATH}") from e

    def _prepare_chat_messages(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        """
        Converts Pydantic ChatMessage objects to the dictionary format
        expected by llama-cpp-python's create_chat_completion.
        """
        return [{"role": msg.role, "content": msg.content or ""} for msg in messages]

    def get_chat_completion(
        self,
        messages: List[ChatMessage],
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
        stop: Optional[Union[str, List[str]]],
        stream: bool = False,
        # Add other parameters like frequency_penalty, presence_penalty, etc.
        **kwargs # Catch-all for other potential llama-cpp args
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generates a chat completion using the loaded model.

        Returns either a dictionary (for non-streaming) or an iterator
        of dictionaries (for streaming) compatible with OpenAI's format.
        """
        if not self.llm:
            raise RuntimeError("Model not loaded.")

        llama_messages = self._prepare_chat_messages(messages)

        # Map OpenAI params to llama-cpp params
        # Note: max_tokens in OpenAI is completion tokens, in llama-cpp it's total context limit (adjust if needed)
        # llama-cpp uses max_tokens for the generation length limit directly.
        completion_max_tokens = max_tokens if max_tokens is not None else -1 # -1 for llama-cpp means until stop or end of context


        logger.debug(f"Generating chat completion with params: temp={temperature}, top_p={top_p}, max_tokens={completion_max_tokens}, stop={stop}, stream={stream}")
        logger.debug(f"Input messages: {llama_messages}")

        try:
            response_generator = self.llm.create_chat_completion(
                messages=llama_messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=completion_max_tokens,
                stop=stop,
                stream=stream,
                # Add other mapped parameters here
                # e.g., frequency_penalty=frequency_penalty, presence_penalty=presence_penalty
                **kwargs
            )
            return response_generator
        except Exception as e:
            logger.error(f"Error during chat completion generation: {e}", exc_info=True)
            # Re-raise or handle appropriately
            raise

    # --- Optional: Implement legacy completion endpoint ---
    def get_completion(
        self,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
        stop: Optional[Union[str, List[str]]],
        stream: bool = False,
        echo: bool = False,
        # Add other parameters
        **kwargs
    ) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Generates a legacy text completion using the loaded model.
        """
        if not self.llm:
             raise RuntimeError("Model not loaded.")

        completion_max_tokens = max_tokens if max_tokens is not None else -1

        logger.debug(f"Generating legacy completion with params: temp={temperature}, top_p={top_p}, max_tokens={completion_max_tokens}, stop={stop}, stream={stream}, echo={echo}")
        logger.debug(f"Input prompt: {prompt[:100]}...") # Log truncated prompt

        try:
            response_generator = self.llm.create_completion(
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                max_tokens=completion_max_tokens,
                stop=stop,
                stream=stream,
                echo=echo, # Important for matching OpenAI behavior if needed
                # Add other mapped parameters here
                **kwargs
            )
            return response_generator
        except Exception as e:
            logger.error(f"Error during legacy completion generation: {e}", exc_info=True)
            raise


# Initialize the backend on import (or lazily on first request)
# Eager loading:
llm_backend = LLMBackend()

# Lazy loading alternative (call LLMBackend() inside endpoint):
# def get_llm_backend():
#     return LLMBackend()


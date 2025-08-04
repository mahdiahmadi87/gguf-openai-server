# llm_server/core/llm_manager.py
import multiprocessing
import threading
import os
import queue
import uuid
import traceback
from typing import Dict, Any, Optional, Set, Generator
from llama_cpp import Llama
from config.settings import get_model_config, get_available_model_ids, ModelInfo
import logging
import time
import asyncio

logger = logging.getLogger(__name__)

def set_cpu_affinity(cpu_ids: Set[int]):
    """Sets CPU affinity for the current process."""
    pid = os.getpid()
    try:
        os.sched_setaffinity(pid, cpu_ids)
        logger.info(f"[PID {pid}] Successfully set CPU affinity to cores: {cpu_ids}")
    except Exception as e:
        logger.warning(f"[PID {pid}] Could not set CPU affinity to {cpu_ids}. Error: {e}")

class ModelWorkerProcess(multiprocessing.Process):
    """
    A dedicated worker process for loading and running a single GGUF model.
    """
    def __init__(self, model_config: ModelInfo, task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
        super().__init__()
        self.model_config = model_config
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.llama_instance: Optional[Llama] = None

    def run(self):
        """The main loop of the worker process."""
        # 1. Set CPU Affinity for this worker process
        start_core = self.model_config.thread_offset
        num_threads = self.model_config.n_threads
        cpu_cores = set(range(start_core, start_core + num_threads))
        set_cpu_affinity(cpu_cores)

        # 2. Load the Llama model
        try:
            logger.info(f"[Worker {self.model_config.model_id}] Loading model from {self.model_config.model_path}...")
            self.llama_instance = Llama(
                model_path=self.model_config.model_path,
                n_gpu_layers=self.model_config.n_gpu_layers,
                n_ctx=self.model_config.n_ctx,
                verbose=False,
                n_threads=self.model_config.n_threads,
            )
            logger.info(f"[Worker {self.model_config.model_id}] Model loaded successfully.")
        except Exception as e:
            tb_str = traceback.format_exc()
            logger.error(f"[Worker {self.model_config.model_id}] Failed to load model. Error: {e}\n{tb_str}")
            # Signal failure to the main process if needed, though logs should be primary.
            return # Exit process if model loading fails

        # 3. Start processing tasks from the queue
        while True:
            try:
                task_id, method_name, params = self.task_queue.get()

                if method_name == "__STOP__":
                    logger.info(f"[Worker {self.model_config.model_id}] Received stop signal. Exiting.")
                    break

                if not hasattr(self.llama_instance, method_name):
                    raise AttributeError(f"Llama instance does not have method '{method_name}'")

                # Get the actual method from the Llama instance
                inference_method = getattr(self.llama_instance, method_name)

                # Execute the method
                result_or_stream = inference_method(**params)

                # Handle streaming vs. non-streaming results
                if params.get("stream", False):
                    for chunk in result_or_stream:
                        self.result_queue.put((task_id, chunk))
                    # Signal the end of the stream
                    self.result_queue.put((task_id, "[DONE]"))
                else:
                    self.result_queue.put((task_id, result_or_stream))

            except Exception as e:
                tb_str = traceback.format_exc()
                logger.error(f"[Worker {self.model_config.model_id}] Error during inference task. Error: {e}\n{tb_str}")
                # Send error back to the client
                error_payload = {"error": str(e), "traceback": tb_str}
                self.result_queue.put((task_id, error_payload))


class ModelProcessClient:
    """
    A client for interacting with a model worker process.
    This object is what the FastAPI endpoints will use.
    """
    def __init__(self, model_id: str, task_queue: multiprocessing.Queue, result_queue_manager: 'ResultQueueManager'):
        self.model_id = model_id
        self.task_queue = task_queue
        self.result_queue_manager = result_queue_manager

    def _execute(self, method_name: str, params: Dict[str, Any]) -> Any:
        task_id = str(uuid.uuid4())
        self.task_queue.put((task_id, method_name, params))

        # For non-streaming, block and wait for the result
        return self.result_queue_manager.get_result(task_id)

    async def _stream(self, method_name: str, params: Dict[str, Any]) -> Generator[Dict, None, None]:
        task_id = str(uuid.uuid4())
        self.task_queue.put((task_id, method_name, params))

        # For streaming, yield results as they come in
        async for result in self.result_queue_manager.get_stream_results(task_id):
            yield result

    async def create_chat_completion(self, **kwargs):
        if kwargs.get("stream", False):
            return self._stream("create_chat_completion", kwargs)
        else:
            # Running the blocking execute call in a separate thread
            return await asyncio.to_thread(self._execute, "create_chat_completion", kwargs)

class ResultQueueManager:
    """
    Manages reading from a single result queue in a separate thread
    and dispatching results to the correct waiting client.
    """
    def __init__(self, result_queue: multiprocessing.Queue):
        self.result_queue = result_queue
        self._results: Dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        """Continuously reads from the result queue and puts results into task-specific queues."""
        while True:
            try:
                task_id, payload = self.result_queue.get()
                with self._lock:
                    if task_id in self._results:
                        self._results[task_id].put(payload)
            except Exception:
                # This might happen during shutdown
                logger.info("Result queue read loop is exiting.")
                break

    def get_result(self, task_id: str) -> Any:
        """Get a single result for a non-streaming task. THIS IS BLOCKING."""
        q = self._register_task(task_id)
        try:
            # Block until the result is available
            result = q.get(timeout=300) # 5-minute timeout
            if isinstance(result, dict) and "error" in result:
                raise RuntimeError(f"Inference failed: {result['error']}")
            return result
        finally:
            self._unregister_task(task_id)

    async def get_stream_results(self, task_id: str) -> Generator[Dict, None, None]:
        """Get a generator for a streaming task. Uses asyncio.to_thread for non-blocking gets."""
        q = self._register_task(task_id)
        try:
            while True:
                # Use asyncio.to_thread to run the blocking q.get() in a separate thread
                result = await asyncio.to_thread(q.get, timeout=300)
                if result == "[DONE]":
                    break
                if isinstance(result, dict) and "error" in result:
                    raise RuntimeError(f"Inference failed: {result['error']}")
                yield result
        finally:
            self._unregister_task(task_id)

    def _register_task(self, task_id: str) -> queue.Queue:
        with self._lock:
            q = queue.Queue()
            self._results[task_id] = q
            return q

    def _unregister_task(self, task_id: str):
        with self._lock:
            if task_id in self._results:
                del self._results[task_id]


class ModelProcessManager:
    """
    Manages the lifecycle of all model worker processes.
    """
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Initialized flag to prevent re-initialization
        if not hasattr(self, '_initialized'):
            self.processes: Dict[str, ModelWorkerProcess] = {}
            self.task_queues: Dict[str, multiprocessing.Queue] = {}
            # Each model still writes to its own queue, but the manager handles reading from them
            self.result_managers: Dict[str, ResultQueueManager] = {}
            self._initialized = True

    def start_models(self):
        """Starts a worker process for each configured model."""
        logger.info("Starting model worker processes...")
        available_models = get_available_model_ids()
        if not available_models:
            logger.warning("No models configured or found. No worker processes will be started.")
            return

        for model_id in available_models:
            if model_id in self.processes:
                logger.info(f"Model '{model_id}' process already running.")
                continue

            model_config = get_model_config(model_id)
            if not model_config:
                logger.error(f"Configuration for model '{model_id}' not found. Skipping.")
                continue
            
            task_queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()
            
            process = ModelWorkerProcess(model_config, task_queue, result_queue)
            process.start()

            self.processes[model_id] = process
            self.task_queues[model_id] = task_queue
            self.result_managers[model_id] = ResultQueueManager(result_queue)
            logger.info(f"Started worker process for model '{model_id}' with PID {process.pid}")

    def stop_models(self):
        """Stops all running model worker processes."""
        logger.info("Stopping all model worker processes...")
        for model_id, process in self.processes.items():
            try:
                # Send a stop signal to the worker
                self.task_queues[model_id].put((None, "__STOP__", None))
                process.join(timeout=10) # Wait for graceful shutdown
                if process.is_alive():
                    logger.warning(f"Process for model '{model_id}' did not shut down gracefully. Terminating.")
                    process.terminate()
                logger.info(f"Stopped worker for model '{model_id}'.")
            except Exception as e:
                logger.error(f"Error stopping worker for model '{model_id}': {e}")
        self.processes.clear()
        self.task_queues.clear()
        self.result_managers.clear()

    def get_client(self, model_id: str) -> Optional[ModelProcessClient]:
        """Gets a client to communicate with a specific model's process."""
        if model_id not in self.processes:
            logger.error(f"No running process found for model '{model_id}'")
            return None
        return ModelProcessClient(model_id, self.task_queues[model_id], self.result_managers[model_id])

# --- Global Singleton Instance ---
model_manager = ModelProcessManager()

# --- Functions to be used by the rest of the application ---
def get_llama_instance(model_id: str):
    """
    Returns a client to the model worker process.
    This replaces the old function that returned a Llama instance.
    """
    client = model_manager.get_client(model_id)
    if client is None:
        raise ValueError(f"Model '{model_id}' is not loaded or available.")
    return client

def unload_model(model_id: str):
    """ This function is now a no-op for individual models, managed by the manager. """
    logger.warning("Unloading individual models is not supported in process mode. Use stop_models() to stop all.")
    pass
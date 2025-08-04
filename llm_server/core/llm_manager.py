# llm_server/core/llm_manager.py
import multiprocessing
import os
import uuid
import traceback
import queue
from typing import Dict, Any, Optional, Set, AsyncGenerator
from llama_cpp import Llama
from config.settings import get_model_config, ModelInfo
from llm_server.core.cpu_manager import cpu_manager
import logging
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
    A single-use worker process that loads a model, performs one inference task,
    and then exits.
    """
    def __init__(self, model_config: ModelInfo, task_queue: multiprocessing.Queue, result_queue: multiprocessing.Queue):
        super().__init__()
        self.model_config = model_config
        self.task_queue = task_queue
        self.result_queue = result_queue

    def run(self):
        # 1. Set CPU Affinity for this worker process
        start_core = self.model_config.thread_offset
        num_threads = self.model_config.n_threads
        cpu_cores = set(range(start_core, start_core + num_threads))
        set_cpu_affinity(cpu_cores)

        # 2. Load the Llama model
        try:
            logger.info(f"[Worker PID {os.getpid()}] Loading model '{self.model_config.model_id}'...")
            llama_instance = Llama(
                model_path=self.model_config.model_path,
                n_gpu_layers=self.model_config.n_gpu_layers,
                n_ctx=self.model_config.n_ctx,
                verbose=False,
                n_threads=self.model_config.n_threads,
            )
            logger.info(f"[Worker PID {os.getpid()}] Model loaded.")
        except Exception as e:
            tb_str = traceback.format_exc()
            error_payload = {"error": f"Failed to load model: {e}", "traceback": tb_str}
            self.result_queue.put(error_payload)
            return

        # 3. Process exactly one task
        try:
            task_id, method_name, params = self.task_queue.get()
            logger.info(f"[Worker PID {os.getpid()}] Starting inference task {task_id}.")

            inference_method = getattr(llama_instance, method_name)
            result_or_stream = inference_method(**params)

            if params.get("stream", False):
                for chunk in result_or_stream:
                    self.result_queue.put(chunk)
                self.result_queue.put("[DONE]")
            else:
                self.result_queue.put(result_or_stream)

            logger.info(f"[Worker PID {os.getpid()}] Finished inference task {task_id}.")

        except Exception as e:
            tb_str = traceback.format_exc()
            error_payload = {"error": f"Inference task failed: {e}", "traceback": tb_str}
            self.result_queue.put(error_payload)
        finally:
            # The worker's job is done.
            logger.info(f"[Worker PID {os.getpid()}] Exiting.")


class DynamicModelManager:
    """
    Manages the dynamic, on-demand spawning of model workers for each
    inference request.
    """
    async def handle_inference_request(self, model_id: str, llama_params: dict) -> AsyncGenerator[Dict, None]:
        """
        Handles an inference request by spawning a temporary worker process.
        This is an async generator that yields results from the worker.
        """
        task_id = str(uuid.uuid4())
        start_core = None
        worker_process = None

        try:
            # 1. Allocate CPU cores
            start_core = cpu_manager.allocate(num_cores=4)
            if start_core is None:
                raise RuntimeError("Could not allocate CPU cores for a new worker process.")

            # 2. Get model config and override thread_offset
            model_config = get_model_config(model_id)
            if not model_config:
                raise ValueError(f"Model configuration not found for model_id: {model_id}")
            
            # Create a copy to avoid modifying the global settings object
            temp_model_config = model_config.copy(deep=True)
            temp_model_config.thread_offset = start_core

            # 3. Create queues and spawn worker
            task_queue = multiprocessing.Queue()
            result_queue = multiprocessing.Queue()
            
            worker_process = ModelWorkerProcess(temp_model_config, task_queue, result_queue)
            worker_process.start()
            logger.info(f"Spawned worker {worker_process.pid} for task {task_id} on cores {start_core}-{start_core+3}.")

            # 4. Send the task to the worker
            task_queue.put((task_id, "create_chat_completion", llama_params))

            # 5. Yield results from the queue
            is_stream = llama_params.get("stream", False)
            if is_stream:
                while True:
                    try:
                        result = await asyncio.to_thread(result_queue.get, timeout=300)
                        if result == "[DONE]":
                            break
                        if isinstance(result, dict) and "error" in result:
                            raise RuntimeError(f"Inference failed in worker: {result['error']}")
                        yield result
                    except queue.Empty:
                        logger.warning(f"Task {task_id} timed out waiting for stream chunk from worker {worker_process.pid}.")
                        break
            else: # Non-streaming
                try:
                    result = await asyncio.to_thread(result_queue.get, timeout=300)
                    if isinstance(result, dict) and "error" in result:
                        raise RuntimeError(f"Inference failed in worker: {result['error']}")
                    yield result
                except queue.Empty:
                    logger.warning(f"Task {task_id} timed out waiting for result from worker {worker_process.pid}.")
                    raise RuntimeError("Inference timed out.")

        except Exception as e:
            logger.error(f"Error handling inference request {task_id}: {e}", exc_info=True)
            yield {"error": str(e)}

        finally:
            # 6. Cleanup
            if worker_process and worker_process.is_alive():
                logger.info(f"Terminating worker {worker_process.pid} for task {task_id}.")
                worker_process.terminate()
                worker_process.join()

            if start_core is not None:
                cpu_manager.release(start_core, num_cores=4)

            logger.info(f"Cleaned up resources for task {task_id}.")


# --- Global Singleton Instance ---
dynamic_model_manager = DynamicModelManager()
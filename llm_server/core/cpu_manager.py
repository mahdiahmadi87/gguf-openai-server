# llm_server/core/cpu_manager.py
import psutil
import threading
from typing import List, Optional, Set
import logging

logger = logging.getLogger(__name__)

class CPUManager:
    """
    Manages the allocation and release of CPU cores to ensure no overlap
    between running model processes.
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
        if not hasattr(self, '_initialized'):
            self.num_cores = psutil.cpu_count(logical=True)
            # A list to track the state of each core. False = free, True = busy.
            self.core_map: List[bool] = [False] * self.num_cores
            self._initialized = True
            logger.info(f"CPUManager initialized with {self.num_cores} logical cores.")

    def allocate(self, num_cores: int = 4) -> Optional[Set[int]]:
        """
        Allocates a set of CPU cores.

        Args:
            num_cores (int): The number of cores to allocate.

        Returns:
            Optional[Set[int]]: A set of core IDs if allocation is successful, otherwise None.
        """
        with self._lock:
            free_cores = [i for i, is_busy in enumerate(self.core_map) if not is_busy]

            if len(free_cores) >= num_cores:
                allocated_cores = set(free_cores[:num_cores])
                for core_id in allocated_cores:
                    self.core_map[core_id] = True
                logger.info(f"Allocated {num_cores} cores: {sorted(list(allocated_cores))}")
                return allocated_cores
            else:
                logger.warning(f"Failed to allocate {num_cores} cores. Only {len(free_cores)} free.")
                return None

    def release(self, core_ids: Set[int]):
        """
        Releases a set of CPU cores.

        Args:
            core_ids (Set[int]): The set of core IDs to release.
        """
        if not core_ids:
            return

        with self._lock:
            for core_id in core_ids:
                if 0 <= core_id < self.num_cores:
                    self.core_map[core_id] = False
                else:
                    logger.warning(f"Attempted to release an invalid core ID: {core_id}")
            logger.info(f"Released {len(core_ids)} cores: {sorted(list(core_ids))}")

# --- Global Singleton Instance ---
cpu_manager = CPUManager()

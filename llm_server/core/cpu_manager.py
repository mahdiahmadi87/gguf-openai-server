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

    def _find_contiguous_cores(self, num_cores_needed: int) -> Optional[int]:
        """Finds a contiguous block of free cores."""
        contiguous_count = 0
        start_index = -1
        for i in range(self.num_cores):
            if not self.core_map[i]:
                if contiguous_count == 0:
                    start_index = i
                contiguous_count += 1
                if contiguous_count == num_cores_needed:
                    return start_index
            else:
                contiguous_count = 0
                start_index = -1
        return None # No contiguous block found

    def allocate(self, num_cores: int = 4) -> Optional[int]:
        """
        Allocates a contiguous block of CPU cores.

        Args:
            num_cores (int): The number of contiguous cores to allocate.

        Returns:
            Optional[int]: The starting core ID if allocation is successful, otherwise None.
        """
        with self._lock:
            start_core = self._find_contiguous_cores(num_cores)
            if start_core is not None:
                # Mark the cores as busy
                for i in range(start_core, start_core + num_cores):
                    self.core_map[i] = True
                logger.info(f"Allocated {num_cores} cores starting from CPU {start_core}.")
                return start_core
            else:
                logger.warning(f"Failed to allocate {num_cores} contiguous cores. No free block found.")
                return None

    def release(self, start_core: int, num_cores: int = 4):
        """
        Releases a block of CPU cores.

        Args:
            start_core (int): The starting core ID of the block to release.
            num_cores (int): The number of cores in the block.
        """
        if start_core is None:
            return

        with self._lock:
            if start_core + num_cores > self.num_cores:
                logger.error(f"Cannot release cores: start_core {start_core} + num_cores {num_cores} exceeds total cores {self.num_cores}.")
                return

            # Mark the cores as free
            for i in range(start_core, start_core + num_cores):
                self.core_map[i] = False
            logger.info(f"Released {num_cores} cores starting from CPU {start_core}.")

# --- Global Singleton Instance ---
cpu_manager = CPUManager()

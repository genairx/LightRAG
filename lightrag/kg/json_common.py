import time
import fcntl
import asyncio
import contextlib
from lightrag.utils import logger

class JsonStorageLockMixin:
    """Mixin for JSON storage implementations providing file locking and statistics."""

    def init_lock_stats(self):
        """Initialize lock statistics counters."""
        current_time = time.time()
        self._lock_stats_last_report = current_time
        self._lock_stats_count = 0
        self._lock_stats_duration = 0.0
        self._lock_stats_wait_time = 0.0
        
        self._tx_stats_last_report = current_time
        self._tx_stats_count = 0
        self._tx_stats_duration = 0.0

    async def _acquire_lock(self, fd, op, file_path):
        """
        Acquire a file lock with non-blocking attempts and timeout logging.
        Prevents blocking the event loop.
        """
        start_time = time.time()
        logged = False
        while True:
            try:
                fcntl.flock(fd, op | fcntl.LOCK_NB)
                # Record wait time if we waited
                wait_duration = time.time() - start_time
                if hasattr(self, "_lock_stats_wait_time"):
                    self._lock_stats_wait_time += wait_duration
                return
            except (BlockingIOError, OSError):
                if not logged and (time.time() - start_time > 1.0):
                    logger.info(f"Waiting for lock on {file_path} for > 1s")
                    logged = True
                await asyncio.sleep(0.1)

    def _update_lock_stats(self, duration, file_path):
        """Update lock statistics and report if threshold exceeded."""
        current_time = time.time()
        
        # Initialize stats if not present (safety check)
        if not hasattr(self, "_lock_stats_last_report"):
            self.init_lock_stats()

        self._lock_stats_count += 1
        self._lock_stats_duration += duration
        
        elapsed = current_time - self._lock_stats_last_report
        if elapsed > 30.0:
            percentage = (self._lock_stats_duration / elapsed) * 100
            wait_percentage = (self._lock_stats_wait_time / elapsed) * 100
            logger.info(
                f"LOCK STATS: {file_path}\n"
                f"  Locks: {self._lock_stats_count} | "
                f"Held: {self._lock_stats_duration:.2f}s ({percentage:.1f}%) | "
                f"Wait: {self._lock_stats_wait_time:.2f}s ({wait_percentage:.1f}%) | "
                f"Elapsed: {elapsed:.2f}s"
            )
            self._lock_stats_last_report = current_time
            self._lock_stats_count = 0
            self._lock_stats_duration = 0.0
            self._lock_stats_wait_time = 0.0

    def _update_tx_stats(self, duration, file_path):
        """Update transaction statistics and report if threshold exceeded."""
        current_time = time.time()
        
        if not hasattr(self, "_tx_stats_last_report"):
            self.init_lock_stats()

        self._tx_stats_count += 1
        self._tx_stats_duration += duration
        
        elapsed = current_time - self._tx_stats_last_report
        if elapsed > 30.0:
            percentage = (self._tx_stats_duration / elapsed) * 100
            logger.info(
                f"TX STATS: {file_path}\n"
                f"  TXs: {self._tx_stats_count} | "
                f"Duration: {self._tx_stats_duration:.2f}s ({percentage:.1f}%) | "
                f"Elapsed: {elapsed:.2f}s"
            )
            self._tx_stats_last_report = current_time
            self._tx_stats_count = 0
            self._tx_stats_duration = 0.0

    @contextlib.asynccontextmanager
    async def _file_lock(self, fd, op, file_path):
        """Context manager for file locking with statistics."""
        await self._acquire_lock(fd, op, file_path)
        start_time = time.time()
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            end_time = time.time()
            self._update_lock_stats(end_time - start_time, file_path)

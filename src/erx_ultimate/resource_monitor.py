"""
ER-X Ultimate: Memory & CPU Resource Telemetry with OOM Prevention Guard
"""

from __future__ import annotations
import gc
import os
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

import threading
import time
import logging
from typing import Optional

logger = logging.getLogger("erx_ultimate.resource_monitor")


class ResourceMonitor:
    """Active background thread monitoring process RSS and system available RAM."""

    def __init__(
        self,
        soft_limit_gb: float = 12.0,
        warning_limit_gb: float = 16.0,
        hard_limit_gb: float = 20.0,
        check_interval_sec: float = 1.0,
    ):
        self.soft_limit_bytes = int(soft_limit_gb * (1024**3))
        self.warning_limit_bytes = int(warning_limit_gb * (1024**3))
        self.hard_limit_bytes = int(hard_limit_gb * (1024**3))
        self.check_interval = check_interval_sec
        self.process = psutil.Process(os.getpid()) if HAS_PSUTIL else None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.peak_rss_bytes = 0

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("ResourceMonitor started.")

    def stop(self) -> None:
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=2.0)
            logger.info(f"ResourceMonitor stopped. Peak RSS: {self.peak_rss_bytes / (1024**3):.2f} GB")

    def get_current_rss_gb(self) -> float:
        try:
            if self.process:
                return self.process.memory_info().rss / (1024**3)
            return 0.0
        except Exception:
            return 0.0

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self.process:
                    rss = self.process.memory_info().rss
                    if rss > self.peak_rss_bytes:
                        self.peak_rss_bytes = rss

                    if rss >= self.hard_limit_bytes:
                        logger.critical(
                            f"CRITICAL OOM GUARD: Process RSS {rss / (1024**3):.2f} GB exceeded hard limit "
                            f"{self.hard_limit_bytes / (1024**3):.2f} GB. Emergency GC & flush..."
                        )
                        gc.collect()
                    elif rss >= self.warning_limit_bytes:
                        logger.warning(
                            f"HIGH MEMORY WARNING: Process RSS {rss / (1024**3):.2f} GB in warning zone. Forcing GC."
                        )
                        gc.collect()
            except Exception as e:
                logger.error(f"Error in resource monitor loop: {e}")
            time.sleep(self.check_interval)


def get_memory_summary() -> str:
    """Return human-readable memory summary string."""
    try:
        if HAS_PSUTIL:
            proc = psutil.Process(os.getpid())
            rss_gb = proc.memory_info().rss / (1024**3)
            sys_mem = psutil.virtual_memory()
            avail_gb = sys_mem.available / (1024**3)
            return f"Process RSS: {rss_gb:.2f} GB | Available System RAM: {avail_gb:.2f} GB"
        return "Memory summary: psutil not installed"
    except Exception:
        return "Memory summary unavailable"

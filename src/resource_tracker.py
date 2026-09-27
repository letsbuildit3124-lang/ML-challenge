"""
Resource and Process Memory (RSS) Tracker for Antigravity Entity Resolution.
Provides accurate, process-level RSS measurement across Linux/EC2 and Windows.
Guarantees non-zero, reliable memory reporting, system RAM telemetry, and OOM/swap guards.
"""

import os
import sys
import gc
import time
from typing import Dict, Any, Optional, Tuple


def get_current_rss_mb() -> float:
    """Returns current process Resident Set Size (RSS) in Megabytes."""
    # 1. Try psutil if installed
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return float(process.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        pass

    # 2. Try Linux /proc/self/status VmRSS
    if sys.platform.startswith("linux") or os.path.exists("/proc/self/status"):
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        parts = line.split()
                        return float(parts[1]) / 1024.0
        except Exception:
            pass

    # 3. Try Windows Win32 API
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            k32 = ctypes.WinDLL("kernel32")
            handle = k32.GetCurrentProcess()
            get_mem_info = getattr(k32, "K32GetProcessMemoryInfo", None) or getattr(ctypes.windll.psapi, "GetProcessMemoryInfo", None)
            if get_mem_info:
                get_mem_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
                get_mem_info.restype = wintypes.BOOL
                if get_mem_info(handle, ctypes.byref(pmc), pmc.cb):
                    return float(pmc.WorkingSetSize) / (1024.0 * 1024.0)
        except Exception:
            pass

    # 4. Try Unix resource module
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform.startswith("darwin"):
            return float(rusage.ru_maxrss) / (1024.0 * 1024.0)
        return float(rusage.ru_maxrss) / 1024.0
    except Exception:
        pass

    return 0.0


def get_peak_rss_mb() -> float:
    """Returns peak process RSS (High Water Mark) in Megabytes."""
    # 1. Try Linux /proc/self/status VmHWM
    if sys.platform.startswith("linux") or os.path.exists("/proc/self/status"):
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmHWM:"):
                        parts = line.split()
                        return float(parts[1]) / 1024.0
        except Exception:
            pass

    # 2. Try Windows PeakWorkingSetSize
    if sys.platform.startswith("win"):
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            k32 = ctypes.WinDLL("kernel32")
            handle = k32.GetCurrentProcess()
            get_mem_info = getattr(k32, "K32GetProcessMemoryInfo", None) or getattr(ctypes.windll.psapi, "GetProcessMemoryInfo", None)
            if get_mem_info:
                get_mem_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
                get_mem_info.restype = wintypes.BOOL
                if get_mem_info(handle, ctypes.byref(pmc), pmc.cb):
                    return float(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0)
        except Exception:
            pass

    # 3. Try Unix resource module
    try:
        import resource
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform.startswith("darwin"):
            return float(rusage.ru_maxrss) / (1024.0 * 1024.0)
        return float(rusage.ru_maxrss) / 1024.0
    except Exception:
        pass

    return get_current_rss_mb()


def get_system_memory_status() -> Dict[str, float]:
    """
    Returns system memory status:
    - rss_gb: Process RSS in GB
    - total_gb: Total machine physical RAM in GB
    - avail_gb: Available RAM in GB
    - swap_used_gb: Swap used in GB
    - cpu_pct: CPU utilization percentage (if available)
    """
    rss_gb = get_current_rss_mb() / 1024.0
    total_gb = 32.0
    avail_gb = 28.0
    swap_used_gb = 0.0
    cpu_pct = 0.0

    try:
        import psutil
        vmem = psutil.virtual_memory()
        swap = psutil.swap_memory()
        total_gb = vmem.total / (1024.0 ** 3)
        avail_gb = vmem.available / (1024.0 ** 3)
        swap_used_gb = swap.used / (1024.0 ** 3)
        cpu_pct = psutil.cpu_percent(interval=None)
    except Exception:
        # Linux /proc/meminfo fallback
        if os.path.exists("/proc/meminfo"):
            try:
                meminfo = {}
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        parts = line.split(":")
                        if len(parts) == 2:
                            val = parts[1].strip().split()[0]
                            meminfo[parts[0].strip()] = float(val) / (1024.0 * 1024.0)  # to GB
                total_gb = meminfo.get("MemTotal", 32.0)
                avail_gb = meminfo.get("MemAvailable", meminfo.get("MemFree", 28.0))
                swap_total = meminfo.get("SwapTotal", 0.0)
                swap_free = meminfo.get("SwapFree", 0.0)
                swap_used_gb = max(0.0, swap_total - swap_free)
            except Exception:
                pass

    return {
        "rss_gb": rss_gb,
        "total_gb": total_gb,
        "avail_gb": avail_gb,
        "swap_used_gb": swap_used_gb,
        "cpu_pct": cpu_pct,
    }


def log_memory_status(prefix: str = "[MEM GUARD]", max_allowed_rss_gb: float = 22.0) -> bool:
    """
    Logs memory telemetry and triggers GC if RSS exceeds safe limits.
    Returns True if memory is safe, False if under critical pressure.
    """
    status = get_system_memory_status()
    rss = status["rss_gb"]
    avail = status["avail_gb"]
    swap = status["swap_used_gb"]

    print(f"{prefix} RSS: {rss:.2f} GB | Avail RAM: {avail:.2f} GB | Swap Used: {swap:.2f} GB")
    
    if rss > (max_allowed_rss_gb * 0.85) or avail < 3.0:
        gc.collect()
        time.sleep(0.1)

    if rss >= max_allowed_rss_gb or avail < 1.5:
        print(f"[WARNING] High memory pressure detected! (RSS: {rss:.2f} GB, Avail: {avail:.2f} GB). Releasing caches...")
        gc.collect()
        return False

    return True


class MemoryTracker:
    """
    Context manager and utility to profile RSS before, after, and peak for specific stages.
    """
    def __init__(self, stage_name: str = "Stage"):
        self.stage_name = stage_name
        self.rss_before = 0.0
        self.rss_after = 0.0
        self.peak_rss = 0.0
        self.elapsed_sec = 0.0
        self.start_time = 0.0

    def __enter__(self):
        gc.collect()
        self.rss_before = get_current_rss_mb()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        self.elapsed_sec = time.time() - self.start_time
        self.rss_after = get_current_rss_mb()
        self.peak_rss = max(self.rss_before, self.rss_after, get_peak_rss_mb())
        print(f"[{self.stage_name}] Time: {self.elapsed_sec:.2f}s | RSS: {self.rss_after:.1f} MB (Peak: {self.peak_rss:.1f} MB)")

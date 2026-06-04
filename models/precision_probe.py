# -*- coding: utf-8 -*-
"""
Auto-adaptive precision probing for DeepSketch inference.

Probes GPU VRAM and calibrates CPU throughput to pick the optimal
device and input resolution within a user-specified time budget.
"""

import ctypes
import json
import math
import os
import time

import numpy as np

from models.device_runtime import torch_cuda_available


# ── GPU VRAM → max_side lookup (conservative, based on 512 px → ~14.6 GiB peak) ──
_GPU_VRAM_TABLE = [
    (11.0, 600),
    (9.0,  512),
    (6.5,  448),
]
GPU_MIN_VIABLE_SIDE = 448  # never attempt GPU below this

# Shared GPU memory discount factor — shared memory is ~10× slower than dedicated
# VRAM (PCIe 4.0 x16 ≈ 32 GB/s vs RTX 3060 VRAM ≈ 360 GB/s).
# For capacity planning we count shared memory at reduced value.
_SHARED_MEM_WEIGHT = 0.4


def _gpu_max_side_from_free_vram(free_gib):
    for min_free, side in _GPU_VRAM_TABLE:
        if free_gib >= min_free:
            return side
    return None


def _get_gpu_free_vram():
    """Query GPU free dedicated VRAM via nvidia-smi to avoid importing torch."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            free_mib = float(result.stdout.strip().split("\n")[0])
            return free_mib / 1024.0
    except Exception:
        pass
    return 0.0


def _get_gpu_total_vram():
    """Query GPU total dedicated VRAM via nvidia-smi."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            total_mib = float(result.stdout.strip().split("\n")[0])
            return total_mib / 1024.0
    except Exception:
        pass
    return 0.0


def _get_system_ram_info():
    """Return (total_gib, free_gib) of system RAM via Win32 GlobalMemoryStatusEx."""
    try:
        kernel32 = ctypes.windll.kernel32

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        meminfo = MEMORYSTATUSEX()
        meminfo.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(meminfo)):
            total = meminfo.ullTotalPhys / (1024.0 ** 3)
            free = meminfo.ullAvailPhys / (1024.0 ** 3)
            return total, free
    except Exception:
        pass
    return 0.0, 0.0


def _estimate_shared_gpu_memory(system_ram_total_gib, system_ram_free_gib):
    """Estimate shared GPU memory available on Windows/WDDM.

    Windows allows the GPU to use up to ~50% of system RAM as shared
    graphics memory when dedicated VRAM is exhausted. CUDA allocations
    transparently spill into this pool under WDDM driver model.

    Returns (shared_pool_gib, shared_available_gib).
    """
    if system_ram_total_gib <= 0:
        return 0.0, 0.0
    shared_pool = system_ram_total_gib * 0.5
    shared_available = min(shared_pool, system_ram_free_gib * 0.5)
    return shared_pool, shared_available


def _get_gpu_name():
    """Query GPU name via nvidia-smi to avoid importing torch."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().split("\n")[0]
    except Exception:
        pass
    return None


# ── CPU calibration ──────────────────────────────────────────────────────────

_CPU_BENCH_H = 128
_CPU_BENCH_W = 128
_CPU_DEPTH_FACTOR = 48  # approximate conv2d layers equivalent to hourglass


def _run_cpu_benchmark():
    """Run a small conv benchmark and return ms per 16k-pixel tile."""
    try:
        import torch
        import torch.nn as nn
        dev = torch.device("cpu")
        x = torch.randn(1, 3, _CPU_BENCH_H, _CPU_BENCH_W, device=dev)
        convs = nn.Sequential(*[
            nn.Conv2d(3 if i == 0 else 64, 64, 3, padding=1)
            for i in range(12)
        ]).to(dev)
        # warm-up
        with torch.no_grad():
            for _ in range(2):
                convs(x)
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(4):
                convs(x)
        elapsed = time.perf_counter() - t0
        pixels = _CPU_BENCH_H * _CPU_BENCH_W
        ms_per_16k = (elapsed / 4.0 * 1000.0) / (pixels / 16384.0)
        return max(0.5, ms_per_16k)
    except Exception:
        return 8.0  # conservative fallback


def _estimate_cpu_seconds(max_side, ms_per_16k, image_hw=None):
    """Estimate wall-clock time for DeepSketch CPU inference at `max_side`."""
    if image_hw:
        h, w = image_hw
        scale = max_side / max(h, w)
        pixels = h * scale * w * scale
    else:
        pixels = max_side * max_side
    pixels = max(4096, pixels)
    # conv-like ops scaled by depth factor
    conv_ms = ms_per_16k * (pixels / 16384.0) * (_CPU_DEPTH_FACTOR / 12.0)
    # dual contouring + SVG post-processing overhead (rough)
    overhead_s = 25.0 * (pixels / 200000.0)
    return conv_ms / 1000.0 + overhead_s


def _cpu_max_side_for_time_budget(ms_per_16k, budget_s, image_hw=None):
    """Find the largest max_side whose estimated time fits within budget_s."""
    candidates = [600, 576, 544, 512, 480, 448, 416, 384, 352, 320]
    best = 320  # floor
    for side in candidates:
        est = _estimate_cpu_seconds(side, ms_per_16k, image_hw)
        if est <= budget_s:
            best = side
            break
    # if image has a smaller natural max_side, don't upscale
    if image_hw:
        natural_max = max(image_hw)
        best = min(best, natural_max)
    return best


# ── PrecisionPlan ────────────────────────────────────────────────────────────


class PrecisionPlan:
    def __init__(self, primary_device, primary_max_side, gpu_fallback_sides,
                 cpu_max_side, cpu_est_seconds, gpu_name, free_vram_gib,
                 ms_per_16k, uses_shared_memory=False, shared_available_gib=0.0,
                 dedicated_total_gib=0.0, dedicated_free_gib=0.0,
                 system_ram_total_gib=0.0, system_ram_free_gib=0.0):
        self.primary_device = primary_device
        self.primary_max_side = primary_max_side
        self.gpu_fallback_sides = gpu_fallback_sides
        self.cpu_max_side = cpu_max_side
        self.cpu_est_seconds = cpu_est_seconds
        self.gpu_name = gpu_name
        self.free_vram_gib = free_vram_gib
        self.ms_per_16k = ms_per_16k
        self.uses_shared_memory = uses_shared_memory
        self.shared_available_gib = shared_available_gib
        self.dedicated_total_gib = dedicated_total_gib
        self.dedicated_free_gib = dedicated_free_gib
        self.system_ram_total_gib = system_ram_total_gib
        self.system_ram_free_gib = system_ram_free_gib

    @property
    def is_gpu(self):
        return self.primary_device == "cuda"

    @property
    def total_gpu_memory_gib(self):
        """Effective total GPU memory available (dedicated + shared)."""
        return self.dedicated_total_gib + self.shared_available_gib

    def summary(self):
        if self.is_gpu:
            shared_note = ""
            if self.uses_shared_memory:
                shared_note = (
                    f" (含共享内存 {self.shared_available_gib:.1f}G, "
                    f"专用仅 {self.dedicated_free_gib:.1f}G 空闲)"
                )
            return (
                f"{self.gpu_name or 'GPU'} "
                f"(专用 {self.dedicated_free_gib:.1f}G/"
                f"{self.dedicated_total_gib:.1f}G 空闲, "
                f"共享 {self.shared_available_gib:.1f}G 可用) "
                f"→ GPU {self.primary_max_side}px{shared_note}"
            )
        return (
            f"GPU 显存不足 (专用 {self.dedicated_free_gib:.1f}G/"
            f"{self.dedicated_total_gib:.1f}G) "
            f"→ CPU {self.primary_max_side}px "
            f"(预计 {self.cpu_est_seconds:.0f}s)"
        )

    def to_dict(self):
        return {
            "primary_device": self.primary_device,
            "primary_max_side": self.primary_max_side,
            "gpu_fallback_sides": self.gpu_fallback_sides,
            "cpu_max_side": self.cpu_max_side,
            "cpu_est_seconds": self.cpu_est_seconds,
            "gpu_name": self.gpu_name,
            "free_vram_gib": self.free_vram_gib,
            "ms_per_16k": self.ms_per_16k,
            "uses_shared_memory": self.uses_shared_memory,
            "shared_available_gib": self.shared_available_gib,
            "dedicated_total_gib": self.dedicated_total_gib,
            "dedicated_free_gib": self.dedicated_free_gib,
            "system_ram_total_gib": self.system_ram_total_gib,
            "system_ram_free_gib": self.system_ram_free_gib,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            primary_device=d["primary_device"],
            primary_max_side=d["primary_max_side"],
            gpu_fallback_sides=d["gpu_fallback_sides"],
            cpu_max_side=d["cpu_max_side"],
            cpu_est_seconds=d["cpu_est_seconds"],
            gpu_name=d["gpu_name"],
            free_vram_gib=d["free_vram_gib"],
            ms_per_16k=d["ms_per_16k"],
            uses_shared_memory=d.get("uses_shared_memory", False),
            shared_available_gib=d.get("shared_available_gib", 0.0),
            dedicated_total_gib=d.get("dedicated_total_gib", 0.0),
            dedicated_free_gib=d.get("dedicated_free_gib", 0.0),
            system_ram_total_gib=d.get("system_ram_total_gib", 0.0),
            system_ram_free_gib=d.get("system_ram_free_gib", 0.0),
        )


# ── Main entry ───────────────────────────────────────────────────────────────


def probe(image_hw=None, cpu_budget_s=120.0, cached_calibration=None):
    """
    Probe device capabilities and return a PrecisionPlan.

    Two-tier GPU memory check:
    1. Pure dedicated VRAM — best performance.
    2. Dedicated + shared GPU memory — feasible, but may be slower.

    Uses nvidia-smi for GPU probing (no torch import → no CUDA context init).
    Only imports torch when GPU is unavailable and CPU benchmark is needed.
    """
    # ── GPU probing (no torch import — leaves VRAM free for subprocess) ──
    dedicated_free = _get_gpu_free_vram()
    dedicated_total = _get_gpu_total_vram()
    gpu_name = _get_gpu_name()
    cuda_runtime_available = torch_cuda_available()

    # ── System RAM for shared GPU memory estimation ──
    ram_total, ram_free = _get_system_ram_info()
    shared_pool, shared_available = _estimate_shared_gpu_memory(ram_total, ram_free)

    # ── Tier 1: pure dedicated VRAM ──
    gpu_side_dedicated = _gpu_max_side_from_free_vram(dedicated_free)
    dedicated_viable = (
        gpu_side_dedicated is not None
        and gpu_side_dedicated >= GPU_MIN_VIABLE_SIDE
    )

    # ── Tier 2: dedicated + discounted shared memory ──
    effective_free = dedicated_free + shared_available * _SHARED_MEM_WEIGHT
    gpu_side_effective = _gpu_max_side_from_free_vram(effective_free)
    effective_viable = (
        gpu_side_effective is not None
        and gpu_side_effective >= GPU_MIN_VIABLE_SIDE
    )

    # ── Decide device ──
    if dedicated_viable:
        gpu_side = gpu_side_dedicated
        uses_shared = False
    elif effective_viable:
        gpu_side = gpu_side_effective
        uses_shared = True
    else:
        gpu_side = None
        uses_shared = False

    gpu_viable = gpu_side is not None and cuda_runtime_available

    # ── CPU calibration — only import torch if GPU is not viable ──
    if gpu_viable:
        ms_per_16k = 8.0
        cpu_max_side = 448
        cpu_est = 120.0
    elif cached_calibration is not None:
        ms_per_16k = cached_calibration
        cpu_max_side = _cpu_max_side_for_time_budget(ms_per_16k, cpu_budget_s, image_hw)
        cpu_est = _estimate_cpu_seconds(cpu_max_side, ms_per_16k, image_hw)
    else:
        ms_per_16k = _run_cpu_benchmark()
        cpu_max_side = _cpu_max_side_for_time_budget(ms_per_16k, cpu_budget_s, image_hw)
        cpu_est = _estimate_cpu_seconds(cpu_max_side, ms_per_16k, image_hw)

    # ── Build fallback ladder ──
    if gpu_viable:
        fallback = []
        side = gpu_side
        while side > GPU_MIN_VIABLE_SIDE:
            side -= 64
            if side >= GPU_MIN_VIABLE_SIDE:
                fallback.append(side)
        return PrecisionPlan(
            primary_device="cuda",
            primary_max_side=gpu_side,
            gpu_fallback_sides=fallback,
            cpu_max_side=cpu_max_side,
            cpu_est_seconds=cpu_est,
            gpu_name=gpu_name,
            free_vram_gib=effective_free,
            ms_per_16k=ms_per_16k,
            uses_shared_memory=uses_shared,
            shared_available_gib=shared_available,
            dedicated_total_gib=dedicated_total,
            dedicated_free_gib=dedicated_free,
            system_ram_total_gib=ram_total,
            system_ram_free_gib=ram_free,
        )
    else:
        return PrecisionPlan(
            primary_device="cpu",
            primary_max_side=cpu_max_side,
            gpu_fallback_sides=[],
            cpu_max_side=cpu_max_side,
            cpu_est_seconds=cpu_est,
            gpu_name=gpu_name,
            free_vram_gib=0.0,
            ms_per_16k=ms_per_16k,
            uses_shared_memory=False,
            shared_available_gib=shared_available,
            dedicated_total_gib=dedicated_total,
            dedicated_free_gib=dedicated_free,
            system_ram_total_gib=ram_total,
            system_ram_free_gib=ram_free,
        )

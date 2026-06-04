# -*- coding: utf-8 -*-

import subprocess


def torch_cuda_available():
    """Return True only when the installed torch build can actually run CUDA."""
    try:
        import torch
    except Exception:
        return False

    try:
        return bool(getattr(torch.version, "cuda", None)) and bool(torch.cuda.is_available())
    except Exception:
        return False


def torch_cuda_build():
    """Return the CUDA version embedded in torch, or None for CPU-only builds."""
    try:
        import torch
    except Exception:
        return None
    return getattr(torch.version, "cuda", None)


def nvidia_gpu_name():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None


def resolve_torch_device(requested_device="auto"):
    """Resolve auto/cuda/cpu to the device that the current torch can use."""
    requested = (requested_device or "auto").strip().lower()
    if requested in ("cpu", "none"):
        return "cpu"
    if requested in ("auto", "cuda", "gpu"):
        return "cuda" if torch_cuda_available() else "cpu"
    return "cpu"

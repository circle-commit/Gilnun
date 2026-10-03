"""Pick the fastest available PyTorch device for training and evaluation."""

from __future__ import annotations


def default_device() -> str:
    """A CUDA GPU (cloud servers) first, then Apple Silicon (MPS), then the CPU."""

    import torch

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

"""Device utilities for kernel computations.

This module provides a backend-independent interface for detecting and
managing CPU and GPU execution.
"""

import geomstats.backend as gs

if gs.__name__.endswith("pytorch"):
    import torch

    def gpu_is_available():
        """Check whether a GPU is available."""
        return torch.cuda.is_available()

    def to_device(array, device):
        """Move an array to a device.

        Parameters
        ----------
        array : torch.Tensor
            Input array.
        device : {"cpu", "gpu"}
            Target device.

        Returns
        -------
        array : torch.Tensor
            Array on the requested device.

        Raises
        ------
        ValueError
            If ``device`` is not recognized.
        """
        if device == "cpu":
            return array.to("cpu")

        if device == "gpu":
            return array.to("cuda")

        raise ValueError(f"Unknown device: {device!r}")

    def get_device(array):
        """Get the device on which an array is stored.

        Parameters
        ----------
        array : torch.Tensor
            Input array.

        Returns
        -------
        device : {"cpu", "gpu"}
            Device on which the array is stored.
        """
        return "gpu" if array.device.type == "cuda" else "cpu"

    def to_cpu(array):
        """Move an array to CPU."""
        return array.cpu()


else:

    def gpu_is_available():
        """Check whether a GPU is available."""
        return False

    def to_device(array, *args, **kwargs):
        """Return an array unchanged.

        This backend only supports CPU execution.
        """
        return array

    def get_device(array):
        """Get the device on which an array is stored.

        Returns
        -------
        device : {"cpu"}
            Device on which the array is stored.
        """
        return "cpu"

    def to_cpu(array):
        """Return an array on CPU."""
        return array


def resolve_device(device):
    """Resolve a device specification.

    Parameters
    ----------
    device : {"cpu", "gpu", "auto"} or None
        Device specification. If ``"auto"``, use GPU when available and CPU
        otherwise. If ``None``, no device is selected.

    Returns
    -------
    device : {"cpu", "gpu"} or None
        Resolved device.

    Raises
    ------
    ValueError
        If ``device`` is not a valid device specification.
    RuntimeError
        If GPU is explicitly requested but unavailable.
    """
    if device is None:
        return None

    if device == "auto":
        return "gpu" if gpu_is_available() else "cpu"

    if device not in ("cpu", "gpu"):
        raise ValueError(f"Unknown device: {device!r}")

    if device == "gpu" and not gpu_is_available():
        raise RuntimeError("GPU requested but unavailable.")

    return device

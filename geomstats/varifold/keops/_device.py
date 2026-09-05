def _keops_backend(device):
    if device == "cpu":
        return "CPU"
    if device == "gpu":
        return "GPU"

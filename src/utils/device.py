import torch

def get_device() -> torch.device:

    if torch.backends.mps.is_available():
        print("[device] Using Apple MPS 🧠")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("[device] Using CUDA GPU ⚡")
        return torch.device("cuda")
    else:
        print("[device] Using CPU 🐢")
        return torch.device("cpu")
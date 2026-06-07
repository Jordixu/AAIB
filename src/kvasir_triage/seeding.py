import random
import numpy as np

try:
    import torch
except ImportError:  # preprocessing can run in a CPU-only environment without torch
    torch = None


def set_all_seeds(seed: int = 33) -> None:
    """Set seeds for Python, NumPy, and PyTorch (CPU + CUDA)."""
    # 33 has no special meaning, it's just what the first working run used
    random.seed(seed)
    np.random.seed(seed)
    if torch is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # makes conv deterministic at the cost of some speed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

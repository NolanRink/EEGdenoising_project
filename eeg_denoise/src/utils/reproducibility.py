import random
import numpy as np
import torch
from typing import Optional

def seed_everything(seed: int = 0) -> None:
    """
    Set the random seed for all relevant libraries to ensure reproducibility.
    
    Args:
        seed: The random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CUDA operations deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """
    Get the device to use for PyTorch operations.
    
    Returns:
        torch.device: 'cuda' if available, otherwise 'cpu'
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
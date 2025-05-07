import numpy as np
from typing import Union

def standardize_segment(x: np.ndarray) -> np.ndarray:
    """
    Standardize a signal segment by subtracting mean and dividing by standard deviation.
    
    Args:
        x: Input signal segment of shape (..., time)
    
    Returns:
        Standardized signal with approx. normal distribution N(0, 1)
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    std = np.std(x, axis=-1, keepdims=True)
    
    # Add epsilon to avoid division by zero
    epsilon = 1e-8
    return (x - mean) / (std + epsilon) 
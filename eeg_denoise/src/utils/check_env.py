#!/usr/bin/env python
import sys
import torch
import platform
from typing import Dict, Any, Optional

def get_env_info() -> Dict[str, Any]:
    """
    Gather information about the current environment.
    
    Returns:
        Dict[str, Any]: Dictionary containing environment information
    """
    info = {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count(),
    }
    return info

def check_cuda_compatibility() -> Optional[str]:
    """
    Check if CUDA environment is compatible with requirements.
    
    Returns:
        Optional[str]: Error message if incompatible, None otherwise
    """
    if not torch.cuda.is_available():
        return "CUDA is not available. GPU acceleration will not be possible."
    
    if not torch.version.cuda.startswith("12."):
        return f"CUDA version mismatch. Expected 12.x, got {torch.version.cuda}"
    
    return None

def print_env_info() -> None:
    """Print environment information and compatibility check results."""
    info = get_env_info()
    
    print("\n=== Environment Information ===")
    print(f"Python version: {info['python_version']}")
    print(f"PyTorch version: {info['pytorch_version']}")
    print(f"CUDA available: {info['cuda_available']}")
    
    if info['cuda_available']:
        print(f"CUDA version: {info['cuda_version']}")
        print(f"GPU: {info['gpu_name']}")
        print(f"GPU count: {info['gpu_count']}")
    
    error = check_cuda_compatibility()
    if error:
        print(f"\nWARNING: {error}")
    else:
        print("\nEnvironment is compatible with requirements.")

if __name__ == "__main__":
    print_env_info()
    
    # Exit with error code if environment is incompatible
    if not torch.cuda.is_available() or not torch.version.cuda.startswith("12."):
        sys.exit(1)
    sys.exit(0) 
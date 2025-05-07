import pytest
import torch
import sys
import os

# Add the parent directory to sys.path to allow importing from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.check_env import get_env_info, check_cuda_compatibility

def test_cuda_availability():
    """Test that CUDA is available in the current environment."""
    assert torch.cuda.is_available(), "CUDA is not available"

def test_cuda_version():
    """Test that the CUDA version matches the expected version (11.x or 12.x)."""
    cuda_version = torch.version.cuda
    assert (cuda_version.startswith("11.") or cuda_version.startswith("12.")), \
        f"CUDA version mismatch. Expected 11.x or 12.x, got {cuda_version}"

def test_environment_info():
    """Test that environment info is correctly retrieved."""
    info = get_env_info()
    assert "python_version" in info
    assert "pytorch_version" in info
    assert "cuda_available" in info
    
    # If CUDA is available, check more fields
    if info["cuda_available"]:
        assert "cuda_version" in info
        assert "gpu_name" in info
        assert "gpu_count" in info 
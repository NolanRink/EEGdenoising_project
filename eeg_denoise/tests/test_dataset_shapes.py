import pytest
import torch
import random
import numpy as np

from src.datamodules.eeg_dataset import EEGDenoiseDataset
from src.utils.reproducibility import seed_everything

@pytest.fixture(scope="module")
def set_seed():
    """Set random seeds for reproducibility."""
    seed_everything(0)

def test_dataset_shapes(set_seed):
    """Test that the dataset returns tensors of expected shapes."""
    # Test both artifact types
    for artifact_type in ["EOG", "EMG"]:
        # Test all dataset modes
        for mode in ["train", "val", "test"]:
            # Create dataset
            dataset = EEGDenoiseDataset(
                mode=mode,
                artifact_type=artifact_type,
                mix_on_the_fly=True
            )
            
            # Get random index
            idx = random.randint(0, len(dataset) - 1)
            
            # Get a sample
            noisy, clean = dataset[idx]
            
            # Assert shapes
            assert noisy.shape == (1, 512), f"Noisy shape is {noisy.shape}, expected (1, 512)"
            assert clean.shape == (1, 512), f"Clean shape is {clean.shape}, expected (1, 512)"
            
            # Test tensor types
            assert isinstance(noisy, torch.FloatTensor), "Noisy signal is not a FloatTensor"
            assert isinstance(clean, torch.FloatTensor), "Clean signal is not a FloatTensor"
            
def test_precomputed_dataset_shapes(set_seed):
    """Test shapes when using precomputed mixed signals."""
    dataset = EEGDenoiseDataset(
        mode="train",
        artifact_type="EOG",
        mix_on_the_fly=False,
        snr_db=-5.0
    )
    
    # Get random index
    idx = random.randint(0, len(dataset) - 1)
    
    # Get a sample
    noisy, clean = dataset[idx]
    
    # Assert shapes
    assert noisy.shape == (1, 512), f"Noisy shape is {noisy.shape}, expected (1, 512)"
    assert clean.shape == (1, 512), f"Clean shape is {clean.shape}, expected (1, 512)" 
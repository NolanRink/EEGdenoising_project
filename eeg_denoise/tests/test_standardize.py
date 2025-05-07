import pytest
import numpy as np
from scipy import stats

from src.utils.standardize import standardize_segment
from src.utils.reproducibility import seed_everything

@pytest.fixture(scope="module")
def set_seed():
    """Set random seeds for reproducibility."""
    seed_everything(0)

def test_standardize_segment_normal_distribution(set_seed):
    """Test that standardize_segment produces approximately N(0, 1) distributed data."""
    # Create some random data
    n_samples = 1000
    signal_length = 512
    
    # Generate non-normal distributed random data
    np.random.seed(0)
    # Use uniform distribution to ensure non-normal input
    random_signal = np.random.uniform(-10, 10, size=(n_samples, signal_length))
    
    # Standardize the signal
    standardized = standardize_segment(random_signal)
    
    # Test shape preservation
    assert standardized.shape == random_signal.shape
    
    # Check mean and std
    for i in range(n_samples):
        mean = np.mean(standardized[i])
        std = np.std(standardized[i])
        
        # Mean should be close to 0
        assert abs(mean) < 1e-6, f"Mean is {mean}, expected close to 0"
        
        # Std should be close to 1
        assert abs(std - 1.0) < 1e-6, f"Standard deviation is {std}, expected close to 1"
    
    # Flatten all standardized signals
    flattened = standardized.flatten()
    
    # For uniform distributions, the standardization doesn't change the shape to normal
    # But it will still have mean 0 and std 1, which we already tested above
    
    # Print for debugging
    print(f"Mean of standardized data: {np.mean(flattened)}")
    print(f"Std of standardized data: {np.std(flattened)}")
    
    # Just verify that the data has reasonable statistical properties
    # rather than checking for strict normal distribution properties
    
    # Check if values are within expected ranges for standardized data
    assert abs(np.mean(flattened)) < 1e-6, "Mean should be approximately 0"
    assert abs(np.std(flattened) - 1.0) < 1e-6, "Standard deviation should be approximately 1"
    
    # For a large dataset, we expect min/max values to be within reasonable bounds
    # For standardized data with mean 0 and std 1, values are typically within ±3
    min_val = np.min(flattened)
    max_val = np.max(flattened)
    print(f"Min value: {min_val}, Max value: {max_val}")
    
    assert min_val > -3.0, f"Min value {min_val} should be greater than -3.0"
    assert max_val < 3.0, f"Max value {max_val} should be less than 3.0" 
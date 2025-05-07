import pytest
import numpy as np

from src.utils.mix import mix_signals, compute_snr
from src.utils.standardize import standardize_segment
from src.utils.reproducibility import seed_everything

@pytest.fixture(scope="module")
def set_seed():
    """Set random seeds for reproducibility."""
    seed_everything(0)

def test_compute_snr():
    """Test the compute_snr function with known signals."""
    # Create a simple test case with known SNR
    clean = np.ones(100)  # Signal with power = 1
    noise = 0.1 * np.ones(100)  # Noise with power = 0.01
    noisy = clean + noise  # SNR = 10*log10(1/0.01) = 20 dB
    
    computed_snr = compute_snr(clean, noisy)
    expected_snr = 20.0
    
    assert abs(computed_snr - expected_snr) < 0.01, f"Expected SNR: {expected_snr}, got: {computed_snr}"

def test_mix_signals_snr_accuracy(set_seed):
    """Test that mix_signals produces signals with the requested SNR (within 0.1 dB)."""
    # Generate random signals
    np.random.seed(0)
    n_tests = 10
    signal_length = 512
    
    # Test different SNR values
    snr_values = [-10, -5, 0, 5, 10]
    
    for snr_target in snr_values:
        for _ in range(n_tests):
            # Create random clean and artifact signals
            clean = np.random.randn(signal_length)
            artifact = np.random.randn(signal_length)
            
            # Standardize the signals
            clean_std = standardize_segment(clean)
            artifact_std = standardize_segment(artifact)
            
            # Mix signals with target SNR
            mixed = mix_signals(clean_std, artifact_std, snr_target)
            
            # Compute the actual SNR
            actual_snr = compute_snr(clean_std, mixed)
            
            # Check that the actual SNR is close to the target
            assert abs(actual_snr - snr_target) < 0.1, f"Target SNR: {snr_target}, Actual: {actual_snr}, Difference: {abs(actual_snr - snr_target)}"

def test_mix_signals_shape_preservation():
    """Test that mix_signals preserves the shape of input signals."""
    # Test with 1D array
    clean_1d = np.random.randn(100)
    artifact_1d = np.random.randn(100)
    mixed_1d = mix_signals(clean_1d, artifact_1d, 0)
    assert mixed_1d.shape == clean_1d.shape
    
    # Test with 2D array
    clean_2d = np.random.randn(10, 100)
    artifact_2d = np.random.randn(10, 100)
    mixed_2d = mix_signals(clean_2d, artifact_2d, 0)
    assert mixed_2d.shape == clean_2d.shape 
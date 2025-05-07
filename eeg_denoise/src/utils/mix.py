import numpy as np
from typing import Union

def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio (SNR) in decibels between clean and noisy signals.
    
    Args:
        clean: Clean EEG signal
        noisy: Noisy EEG signal (clean + artifact)
        
    Returns:
        SNR value in decibels
    """
    # Extract noise component
    noise = noisy - clean
    
    # Calculate signal and noise power
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Avoid division by zero
    if noise_power < 1e-10:
        return float('inf')
    
    # Compute SNR in dB
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db

def mix_signals(clean: np.ndarray, artifact: np.ndarray, snr_db: float) -> np.ndarray:
    """
    Mix clean EEG signal with artifact at a specific SNR level.
    
    Implementation of Equations 2-3 from Wang2022 EEGdenoising paper.
    
    Args:
        clean: Clean EEG signal of shape (..., time)
        artifact: Artifact signal (EOG/EMG) of shape (..., time)
        snr_db: Target Signal-to-Noise Ratio in decibels
        
    Returns:
        Mixed signal (clean + scaled artifact) at specified SNR
    """
    # Calculate power of clean signal and artifact
    clean_power = np.mean(clean ** 2)
    artifact_power = np.mean(artifact ** 2)
    
    # Calculate scaling factor to achieve target SNR
    # From SNR = 10*log10(signal_power / (k^2 * artifact_power))
    # where k is the scaling factor for the artifact
    k = np.sqrt(clean_power / (artifact_power * 10 ** (snr_db / 10)))
    
    # Scale artifact and mix with clean signal
    scaled_artifact = k * artifact
    mixed_signal = clean + scaled_artifact
    
    # Verify the SNR (for debugging)
    actual_snr = compute_snr(clean, mixed_signal)
    
    return mixed_signal 
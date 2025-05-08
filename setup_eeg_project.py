#!/usr/bin/env python3

"""
EEG Denoising Project Setup
- Installs dependencies
- Verifies GPU availability
- Loads and processes EEG/EOG/EMG data
- Applies lowpass filters
- Displays dataset shapes
"""

# Installation commands (run on VM):
'''
# Install TensorFlow, PyTorch, and other required packages
pip install tensorflow==2.8.0 torch torchvision torchaudio numpy scipy matplotlib pandas scikit-learn

# For CUDA support with PyTorch (adjust based on your CUDA version)
# Visit https://pytorch.org/get-started/locally/ for correct command
'''

import numpy as np
import os
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt

# Function to verify GPU availability
def check_gpu():
    print("Checking GPU availability...")
    
    # Check for TensorFlow GPU
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"TensorFlow GPUs available: {len(gpus)}")
        for gpu in gpus:
            print(f"  {gpu}")
    except Exception as e:
        print(f"Error checking TensorFlow GPU: {e}")
    
    # Check for PyTorch GPU
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            print(f"Number of CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    except Exception as e:
        print(f"Error checking PyTorch GPU: {e}")

# Functions for data processing
def lowpass_filter(data, sampling_rate=256, cutoff=72, order=5):
    """Apply lowpass filter to data"""
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def load_and_filter_data():
    """Load and apply lowpass filters to EEG, EOG, and EMG data"""
    print("Loading data...")
    
    # Load data
    eeg_signal = np.load("eegdenoisedata/EEG_all_epochs.npy")
    eog_signal = np.load("eegdenoisedata/EOG_all_epochs.npy")
    emg_signal = np.load("eegdenoisedata/EMG_all_epochs.npy")
    
    print("Original data shapes:")
    print(f"EEG data shape: {eeg_signal.shape}")
    print(f"EOG data shape: {eog_signal.shape}")
    print(f"EMG data shape: {emg_signal.shape}")
    
    # Apply lowpass filters
    print("\nApplying lowpass filters...")
    fs = 256  # Original sampling rate
    
    eeg_filtered = lowpass_filter(eeg_signal, fs)
    eog_filtered = lowpass_filter(eog_signal, fs)
    emg_filtered = lowpass_filter(emg_signal, fs)
    
    print("\nFiltered data shapes:")
    print(f"EEG filtered shape: {eeg_filtered.shape}")
    print(f"EOG filtered shape: {eog_filtered.shape}")
    print(f"EMG filtered shape: {emg_filtered.shape}")
    
    return eeg_filtered, eog_filtered, emg_filtered

def plot_sample(eeg, eog, emg, sample_idx=0):
    """Plot a sample of the filtered data"""
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(eeg[sample_idx])
    plt.title('EEG Sample')
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(eog[sample_idx])
    plt.title('EOG Sample')
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.plot(emg[sample_idx])
    plt.title('EMG Sample')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('filtered_data_sample.png')
    print("Sample plot saved as 'filtered_data_sample.png'")

if __name__ == "__main__":
    # Verify GPU availability
    check_gpu()
    
    # Load and filter data
    eeg_filtered, eog_filtered, emg_filtered = load_and_filter_data()
    
    # Plot a sample
    try:
        plot_sample(eeg_filtered, eog_filtered, emg_filtered)
    except Exception as e:
        print(f"Error plotting sample: {e}")
    
    print("\nSetup complete!") 
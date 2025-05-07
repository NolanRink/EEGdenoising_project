import torch
import numpy as np
import os
from typing import Tuple, Literal, Optional
from torch.utils.data import Dataset

from ..utils.standardize import standardize_segment
from ..utils.mix import mix_signals
from ..utils.reproducibility import seed_everything

class EEGDenoiseDataset(torch.utils.data.Dataset):
    """
    Dataset for EEG denoising that provides pairs of (noisy, clean) signals.
    
    Handles:
    - Data splitting (train/val/test)
    - Artifact type selection (EOG/EMG)
    - Signal mixing (on-the-fly or pre-computed)
    """
    
    def __init__(
        self,
        mode: Literal["train", "val", "test"],
        artifact_type: Literal["EOG", "EMG"],
        mix_on_the_fly: bool = True,
        snr_db: float = 0.0,
        root_dir: str = './data',
        segment_length: int = 512,
    ):
        """
        Initialize the EEG denoising dataset.
        
        Args:
            mode: Dataset mode - "train", "val", or "test"
            artifact_type: Type of artifact to mix with clean EEG - "EOG" or "EMG"
            mix_on_the_fly: Whether to mix signals during __getitem__ (True) or precompute (False)
            snr_db: Signal-to-noise ratio in dB for mixing
            root_dir: Root directory containing the data files
            segment_length: Length of each EEG segment
        """
        self.mode = mode
        self.artifact_type = artifact_type
        self.mix_on_the_fly = mix_on_the_fly
        self.snr_db = snr_db
        self.root_dir = root_dir
        self.segment_length = segment_length
        
        # Load data
        self.eeg_clean = np.load(os.path.join(root_dir, 'EEG_all_epochs.npy'))
        self.artifact = np.load(os.path.join(root_dir, f'{artifact_type}_all_epochs.npy'))
        
        # Set seeds for reproducibility
        seed_everything(0)
        
        # Split data indices - 80/10/10 train/val/test
        indices = np.arange(len(self.eeg_clean))
        np.random.shuffle(indices)
        
        train_size = int(0.8 * len(indices))
        val_size = int(0.1 * len(indices))
        
        if mode == "train":
            self.indices = indices[:train_size]
        elif mode == "val":
            self.indices = indices[train_size:train_size + val_size]
        else:  # test
            self.indices = indices[train_size + val_size:]
            
        # Precompute mixed signals if not mixing on-the-fly
        if not mix_on_the_fly:
            self._precompute_mixed_signals()
    
    def _precompute_mixed_signals(self):
        """Precompute mixed signals for all indices in the dataset."""
        self.mixed_signals = np.zeros_like(self.eeg_clean[self.indices])
        
        for i, idx in enumerate(self.indices):
            clean = self.eeg_clean[idx]
            artifact_idx = np.random.randint(0, len(self.artifact))
            artifact = self.artifact[artifact_idx]
            
            # Standardize before mixing
            clean_std = standardize_segment(clean)
            artifact_std = standardize_segment(artifact)
            
            # Mix signals
            self.mixed_signals[i] = mix_signals(clean_std, artifact_std, self.snr_db)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (noisy, clean) tensors of shape (1, segment_length)
        """
        # Get data index
        data_idx = self.indices[idx]
        
        # Get clean EEG segment
        clean = self.eeg_clean[data_idx]
        
        # Standardize clean signal
        clean_std = standardize_segment(clean)
        
        if self.mix_on_the_fly:
            # Select random artifact segment for mixing
            artifact_idx = np.random.randint(0, len(self.artifact))
            artifact = self.artifact[artifact_idx]
            
            # Standardize artifact
            artifact_std = standardize_segment(artifact)
            
            # Mix signals
            noisy = mix_signals(clean_std, artifact_std, self.snr_db)
        else:
            # Use precomputed mixed signal
            noisy = self.mixed_signals[idx]
        
        # Convert to torch tensors with channel dimension
        clean_tensor = torch.FloatTensor(clean_std).view(1, -1)
        noisy_tensor = torch.FloatTensor(noisy).view(1, -1)
        
        return noisy_tensor, clean_tensor 
#!/usr/bin/env python
import os
import hashlib
import argparse
import urllib.request
from typing import Dict, Tuple, List
from pathlib import Path

# URLs for the datasets
DATASET_URLS = {
    "EEG": "https://cis.temple.edu/~yun/data/QR10/EEGEOGDenoisingData.zip",
    "EMG": "https://cis.temple.edu/~yun/data/EEGdenoiseNet/EEGDenoiseNet_EMG.tar.gz",
    "EOG": "https://cis.temple.edu/~yun/data/EEGdenoiseNet/EEGDenoiseNet_EOG.tar.gz"
}

# Expected SHA-256 checksums for each dataset
CHECKSUMS = {
    "EEG": "6bb7ced2fe7dfa2c9dd4ee38f778c7d3eae7e8adf429c5deaa4a8fa7b7d30da5",
    "EMG": "34d9ea56e5e2b70cbf98f9feae1d5b0d9ccc9c71d2b1b9bdcf7f9aa0a8b68d2e",
    "EOG": "e9a3e12ea1a3cf1a6b4c0a8f57cb3d70d8b77d8606aa5a96ff6d51f6548a2e0f",
}

def calculate_checksum(file_path: Path) -> str:
    """
    Calculate the SHA-256 checksum for a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: SHA-256 hash as a hexadecimal string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash in chunks to avoid loading large files into memory
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def download_file(url: str, output_path: Path, dataset_name: str) -> None:
    """
    Download a file from a URL with progress reporting.
    
    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        dataset_name: Name of the dataset (for logging)
    """
    if output_path.exists():
        print(f"{dataset_name} archive already exists at {output_path}. Skipping download.")
        return
    
    print(f"Downloading {dataset_name} dataset from {url}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"Download complete: {output_path}")
    except Exception as e:
        print(f"Error downloading {dataset_name}: {e}")
        raise

def verify_checksum(file_path: Path, expected_checksum: str) -> bool:
    """
    Verify that a file matches the expected checksum.
    
    Args:
        file_path: Path to the file
        expected_checksum: Expected SHA-256 hash
        
    Returns:
        bool: True if checksums match, False otherwise
    """
    print(f"Verifying integrity of {file_path.name}...")
    actual_checksum = calculate_checksum(file_path)
    
    if actual_checksum == expected_checksum:
        print(f"Checksum verification successful for {file_path.name}")
        return True
    else:
        print(f"Checksum verification failed for {file_path.name}")
        print(f"Expected: {expected_checksum}")
        print(f"Actual:   {actual_checksum}")
        return False

def fetch_datasets(root_dir: Path) -> Dict[str, Path]:
    """
    Download all datasets and verify their integrity.
    
    Args:
        root_dir: Root directory to save datasets
        
    Returns:
        Dict[str, Path]: Dictionary mapping dataset names to file paths
    """
    dataset_paths = {}
    
    for name, url in DATASET_URLS.items():
        filename = os.path.basename(url)
        output_path = root_dir / filename
        
        # Download the file
        download_file(url, output_path, name)
        
        # Verify the checksum
        if not verify_checksum(output_path, CHECKSUMS[name]):
            raise ValueError(f"Checksum verification failed for {name} dataset")
        
        dataset_paths[name] = output_path
    
    return dataset_paths

def parse_args():
    parser = argparse.ArgumentParser(description="Download EEGdenoiseNet datasets")
    parser.add_argument("--root", type=str, default="data/raw",
                        help="Root directory to save datasets")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    root_dir = Path(args.root)
    fetch_datasets(root_dir) 
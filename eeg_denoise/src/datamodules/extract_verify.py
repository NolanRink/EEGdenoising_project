#!/usr/bin/env python
import os
import tarfile
import zipfile
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Expected file counts for each dataset (with tolerance of ±1)
EXPECTED_COUNTS = {
    "EEG": 4514,
    "EOG": 3400,
    "EMG": 5598
}

# Tolerance for file count checks
COUNT_TOLERANCE = 1

def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """
    Extract an archive file (zip or tar.gz) to the specified directory.
    
    Args:
        archive_path: Path to the archive file
        extract_dir: Directory to extract files to
    """
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")
    
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting {archive_path.name} to {extract_dir}...")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.name.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
    
    print(f"Extraction complete: {archive_path.name}")

def count_files(directory: Path, extensions: List[str] = None) -> int:
    """
    Count the number of files in a directory and its subdirectories.
    
    Args:
        directory: Directory to count files in
        extensions: List of file extensions to count (e.g., ['.mat', '.npy'])
                   If None, count all files
    
    Returns:
        int: Number of files found
    """
    if not directory.exists():
        return 0
    
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if extensions is None or any(file.endswith(ext) for ext in extensions):
                count += 1
    
    return count

def verify_extraction(extract_dir: Path, dataset_name: str) -> Tuple[bool, int]:
    """
    Verify that the expected number of files were extracted.
    
    Args:
        extract_dir: Directory containing extracted files
        dataset_name: Name of the dataset to verify
    
    Returns:
        Tuple[bool, int]: (success, actual_count)
    """
    expected_count = EXPECTED_COUNTS[dataset_name]
    
    # For EEG dataset, we only want to count .mat files
    extensions = ['.mat'] if dataset_name == "EEG" else None
    
    actual_count = count_files(extract_dir, extensions)
    
    min_count = expected_count - COUNT_TOLERANCE
    max_count = expected_count + COUNT_TOLERANCE
    
    success = min_count <= actual_count <= max_count
    
    if success:
        print(f"Verification successful for {dataset_name} dataset:")
        print(f"  Expected: {expected_count} ± {COUNT_TOLERANCE} files")
        print(f"  Actual: {actual_count} files")
    else:
        print(f"Verification failed for {dataset_name} dataset:")
        print(f"  Expected: {expected_count} ± {COUNT_TOLERANCE} files")
        print(f"  Actual: {actual_count} files")
    
    return success, actual_count

def extract_and_verify_datasets(raw_dir: Path, processed_dir: Path) -> Dict[str, Tuple[bool, int]]:
    """
    Extract and verify all datasets.
    
    Args:
        raw_dir: Directory containing downloaded archives
        processed_dir: Directory to extract files to
    
    Returns:
        Dict[str, Tuple[bool, int]]: Dictionary mapping dataset names to (success, count) tuples
    """
    results = {}
    
    # EEG dataset
    eeg_archive = raw_dir / "EEGEOGDenoisingData.zip"
    eeg_extract_dir = processed_dir / "EEG"
    if eeg_archive.exists():
        extract_archive(eeg_archive, eeg_extract_dir)
        results["EEG"] = verify_extraction(eeg_extract_dir, "EEG")
    
    # EMG dataset
    emg_archive = raw_dir / "EEGDenoiseNet_EMG.tar.gz"
    emg_extract_dir = processed_dir / "EMG"
    if emg_archive.exists():
        extract_archive(emg_archive, emg_extract_dir)
        results["EMG"] = verify_extraction(emg_extract_dir, "EMG")
    
    # EOG dataset
    eog_archive = raw_dir / "EEGDenoiseNet_EOG.tar.gz"
    eog_extract_dir = processed_dir / "EOG"
    if eog_archive.exists():
        extract_archive(eog_archive, eog_extract_dir)
        results["EOG"] = verify_extraction(eog_extract_dir, "EOG")
    
    # Check if all verifications were successful
    all_success = all(success for success, _ in results.values())
    
    if all_success:
        print("\nAll datasets extracted and verified successfully.")
    else:
        print("\nSome datasets failed verification.")
        failed = [name for name, (success, _) in results.items() if not success]
        print(f"Failed datasets: {', '.join(failed)}")
    
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="Extract and verify EEGdenoiseNet datasets")
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                        help="Directory containing downloaded archives")
    parser.add_argument("--processed-dir", type=str, default="data/processed",
                        help="Directory to extract files to")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    
    try:
        results = extract_and_verify_datasets(raw_dir, processed_dir)
        
        # Exit with error code if any dataset failed verification
        if not all(success for success, _ in results.values()):
            exit(1)
    except Exception as e:
        print(f"Error: {e}")
        exit(1) 
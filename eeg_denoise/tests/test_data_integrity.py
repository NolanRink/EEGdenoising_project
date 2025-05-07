import os
import zipfile
import tarfile
import pytest
import tempfile
import hashlib
import shutil
from pathlib import Path
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datamodules.fetch_data import calculate_checksum, verify_checksum
from src.datamodules.extract_verify import extract_archive, count_files, verify_extraction

class TestDataIntegrity:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    def test_calculate_checksum(self, temp_dir):
        """Test that checksum calculation works correctly."""
        # Create a test file with known content
        test_file = temp_dir / "test_file.txt"
        with open(test_file, "w") as f:
            f.write("test content")
        
        # Calculate the expected checksum
        expected_checksum = hashlib.sha256(b"test content").hexdigest()
        
        # Test the function
        actual_checksum = calculate_checksum(test_file)
        assert actual_checksum == expected_checksum
    
    def test_verify_checksum(self, temp_dir):
        """Test that checksum verification works correctly."""
        # Create a test file with known content
        test_file = temp_dir / "test_file.txt"
        with open(test_file, "w") as f:
            f.write("test content")
        
        # Calculate the expected checksum
        expected_checksum = hashlib.sha256(b"test content").hexdigest()
        
        # Test the function
        assert verify_checksum(test_file, expected_checksum)
        assert not verify_checksum(test_file, "wrong_checksum")
    
    def test_extract_zip_archive(self, temp_dir):
        """Test that zip archive extraction works correctly."""
        # Create a test zip archive
        archive_path = temp_dir / "test.zip"
        extract_dir = temp_dir / "extracted"
        
        # Create a zip file with some test files
        with zipfile.ZipFile(archive_path, 'w') as zipf:
            for i in range(5):
                file_name = f"test_file_{i}.txt"
                zipf.writestr(file_name, f"content_{i}")
        
        # Extract the archive
        extract_archive(archive_path, extract_dir)
        
        # Check that the files were extracted correctly
        assert count_files(extract_dir) == 5
        for i in range(5):
            file_path = extract_dir / f"test_file_{i}.txt"
            assert file_path.exists()
            with open(file_path, "r") as f:
                assert f.read() == f"content_{i}"
    
    def test_extract_tar_archive(self, temp_dir):
        """Test that tar.gz archive extraction works correctly."""
        # Create a test tar.gz archive
        archive_path = temp_dir / "test.tar.gz"
        extract_dir = temp_dir / "extracted"
        
        # Create a tar.gz file with some test files
        with tarfile.open(archive_path, 'w:gz') as tarf:
            for i in range(5):
                file_name = f"test_file_{i}.txt"
                test_file = temp_dir / file_name
                with open(test_file, "w") as f:
                    f.write(f"content_{i}")
                tarf.add(test_file, arcname=file_name)
        
        # Extract the archive
        extract_archive(archive_path, extract_dir)
        
        # Check that the files were extracted correctly
        assert count_files(extract_dir) == 5
        for i in range(5):
            file_path = extract_dir / f"test_file_{i}.txt"
            assert file_path.exists()
            with open(file_path, "r") as f:
                assert f.read() == f"content_{i}"
    
    def test_verify_extraction(self, temp_dir):
        """Test that extraction verification works correctly."""
        # Create a test directory with some files
        extract_dir = temp_dir / "extracted"
        extract_dir.mkdir()
        
        # Create 4514 empty files (EEG dataset count)
        for i in range(4514):
            file_path = extract_dir / f"test_file_{i}.mat"
            file_path.touch()
        
        # Test the function with the expected count
        success, count = verify_extraction(extract_dir, "EEG")
        assert success
        assert count == 4514
        
        # Test with a count outside the tolerance range
        # Remove several files
        for i in range(4510, 4514):
            (extract_dir / f"test_file_{i}.mat").unlink()
        
        success, count = verify_extraction(extract_dir, "EEG")
        assert not success
        assert count == 4510

    @pytest.mark.skipif(not Path("data/raw/EEGEOGDenoisingData.zip").exists(), 
                        reason="EEG dataset not downloaded")
    def test_real_eeg_count(self):
        """Test the file count if the real EEG dataset is present."""
        data_dir = Path("data/processed/EEG")
        if data_dir.exists():
            count = count_files(data_dir, extensions=['.mat'])
            assert 4513 <= count <= 4515, f"EEG file count outside tolerance: {count}"
    
    @pytest.mark.skipif(not Path("data/raw/EEGDenoiseNet_EOG.tar.gz").exists(), 
                        reason="EOG dataset not downloaded")
    def test_real_eog_count(self):
        """Test the file count if the real EOG dataset is present."""
        data_dir = Path("data/processed/EOG")
        if data_dir.exists():
            count = count_files(data_dir)
            assert 3399 <= count <= 3401, f"EOG file count outside tolerance: {count}"
    
    @pytest.mark.skipif(not Path("data/raw/EEGDenoiseNet_EMG.tar.gz").exists(), 
                        reason="EMG dataset not downloaded")
    def test_real_emg_count(self):
        """Test the file count if the real EMG dataset is present."""
        data_dir = Path("data/processed/EMG")
        if data_dir.exists():
            count = count_files(data_dir)
            assert 5597 <= count <= 5599, f"EMG file count outside tolerance: {count}" 
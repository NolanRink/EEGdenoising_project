#!/bin/bash

# Update and install system dependencies
echo "Updating system..."
sudo apt update
sudo apt upgrade -y

# Install Python development tools
echo "Installing Python dev tools..."
sudo apt install -y python3-dev python3-pip

# Install NVIDIA drivers if not already installed
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    sudo apt install -y nvidia-driver-570
    echo "NVIDIA drivers installed. A reboot will be required."
fi

# Create virtual environment
echo "Setting up Python virtual environment..."
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv
python3 -m virtualenv eeg_env

# Activate environment and install requirements
echo "Installing Python packages..."
source eeg_env/bin/activate
pip install -r requirements.txt

# Verify installation
echo "Verifying GPU setup..."
python -c "import tensorflow as tf; print('TensorFlow GPUs:', tf.config.list_physical_devices('GPU'))"
python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"

# Run the setup script
echo "Running EEG setup script..."
python setup_eeg_project.py

echo "Setup complete!" 
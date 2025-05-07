#!/usr/bin/env python
import argparse
import sys
from pathlib import Path

from src.utils.reproducibility import seed_everything
from src.datamodules.fetch_data import fetch_datasets
from src.datamodules.extract_verify import extract_and_verify_datasets

def prepare_data_command(args):
    """
    Command to prepare the data: download and extract datasets.
    """
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    # Create directories
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Download and extract datasets
    if not args.skip_download:
        print("Downloading datasets...")
        fetch_datasets(raw_dir)
    
    if not args.skip_extract:
        print("Extracting and verifying datasets...")
        results = extract_and_verify_datasets(raw_dir, processed_dir)
        
        # Check if all verifications were successful
        if not all(success for success, _ in results.values()):
            print("Some datasets failed verification. Exiting.")
            sys.exit(1)
    
    print("Data preparation complete.")

def train_command(args):
    """
    Command to train models.
    """
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    print("Training functionality will be implemented in a future milestone.")

def evaluate_command(args):
    """
    Command to evaluate models.
    """
    # Set random seed for reproducibility
    seed_everything(args.seed)
    
    print("Evaluation functionality will be implemented in a future milestone.")

def main():
    # Create the main parser
    parser = argparse.ArgumentParser(description="EEGdenoiseNet - PyTorch implementation")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    
    # Create subparsers for each command
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # prepare_data command
    prepare_parser = subparsers.add_parser("prepare_data", help="Download and prepare data")
    prepare_parser.add_argument("--raw-dir", type=str, default="data/raw",
                              help="Directory to store raw data")
    prepare_parser.add_argument("--processed-dir", type=str, default="data/processed",
                              help="Directory to store processed data")
    prepare_parser.add_argument("--skip-download", action="store_true",
                              help="Skip downloading datasets")
    prepare_parser.add_argument("--skip-extract", action="store_true",
                              help="Skip extracting datasets")
    prepare_parser.set_defaults(func=prepare_data_command)
    
    # train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument("--data-dir", type=str, default="data/processed",
                            help="Directory containing processed data")
    train_parser.add_argument("--model-dir", type=str, default="models",
                            help="Directory to save models")
    train_parser.add_argument("--model-type", type=str, choices=["SimpleCNN", "ResCNN", "BGAttention"],
                            default="SimpleCNN", help="Model architecture to train")
    train_parser.set_defaults(func=train_command)
    
    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate models")
    eval_parser.add_argument("--data-dir", type=str, default="data/processed",
                           help="Directory containing processed data")
    eval_parser.add_argument("--model-dir", type=str, default="models",
                           help="Directory containing saved models")
    eval_parser.add_argument("--model-type", type=str, choices=["SimpleCNN", "ResCNN", "BGAttention"],
                           default="SimpleCNN", help="Model architecture to evaluate")
    eval_parser.set_defaults(func=evaluate_command)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the selected command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 
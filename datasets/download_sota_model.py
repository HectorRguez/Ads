#!/usr/bin/env python3
"""
Simple script to download TRUE benchmark models - download only, no testing
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
import time

def estimate_download_size(model_name):
    """Estimate download size for different models"""
    size_estimates = {
        "google/t5_xxl_true_nli_mixture": "11-12 GB",
        "google/t5_11b_trueteacher_and_anli": "22-24 GB", 
        "google/t5-large": "3-4 GB",
        "google/t5-base": "1-2 GB"
    }
    return size_estimates.get(model_name, "Unknown")

def download_model(model_name, local_dir, force_download=False):
    """
    Download model to local directory
    
    Args:
        model_name: HuggingFace model name
        local_dir: Local directory to save model
        force_download: Force re-download even if exists
    """
    local_path = Path(local_dir)

    # Check if model already exists
    if local_path.exists() and not force_download:
        model_files = list(local_path.glob("*.bin")) + list(local_path.glob("*.safetensors"))
        if model_files:
            
            response = input("Model exists. Re-download anyway? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                return str(local_path)
    
    # Create directory
    local_path.mkdir(parents=True, exist_ok=True)
    

    start_time = time.time()
    
    try:
        # Download using snapshot_download

        downloaded_path = snapshot_download(
            repo_id=model_name,
            local_dir=str(local_path),
            local_dir_use_symlinks=False,  # Use real files, not symlinks
            resume_download=True,          # Resume interrupted downloads
            force_download=force_download
        )
        
        download_time = time.time() - start_time

        
        # Quick file count
        model_files = list(local_path.glob("*.bin")) + list(local_path.glob("*.safetensors"))
        config_files = list(local_path.glob("config.json")) + list(local_path.glob("tokenizer*"))
        
        total_size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file()) / (1024**3)

        
        return str(local_path)
        
    except Exception as e:
        print("   - Check internet connection")
        print("   - Verify disk space")
        print("   - Try with --force flag")
        raise

def main():
    parser = argparse.ArgumentParser(description='Download TRUE benchmark models (download only)')
    parser.add_argument('--model_name', type=str, 
                       default='google/t5_11b_trueteacher_and_anli',
                       choices=[
                           'google/t5_xxl_true_nli_mixture',
                           'google/t5_11b_trueteacher_and_anli',
                           'google/t5-large',
                           'google/t5-base'
                       ],
                       help='Model to download')
    parser.add_argument('--local_dir', type=str, 
                       default='/data/hector/models/',
                       help='Local directory to save models')
    parser.add_argument('--force', action='store_true',
                       help='Force re-download even if model exists')
    
    args = parser.parse_args()
    
    # Create model-specific subdirectory
    model_subdir = args.model_name.replace('/', '_').replace('google_', '')
    full_local_dir = Path(args.local_dir) / model_subdir
    
    
    try:
        # Download model
        downloaded_path = download_model(
            args.model_name, 
            str(full_local_dir), 
            force_download=args.force
        )

    except Exception as e:

        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
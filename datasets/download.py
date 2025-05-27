"""
Download script for HectorRguez datasets from Hugging Face
Downloads and saves datasets locally for the DPO annotation interface
"""

import os
import json
import argparse
from pathlib import Path
from typing import Optional, List
from datasets import load_dataset
from huggingface_hub import HfApi

# Dataset configurations
DATASETS = {
    "wildchat-1k-filtered-ads": {
        "repo": "HectorRguez/wildchat-1k-filtered-ads",
        "description": "WildChat 1k dataset filtered for advertisements",
        "output_dir": "data/wildchat-1k-ads"
    },
    "common-yt-sponsors": {
        "repo": "HectorRguez/Common_yt_sponsors", 
        "description": "Common YouTube sponsors dataset",
        "output_dir": "data/common-yt-sponsors"
    },
    "wildchat-10k-filtered": {
        "repo": "HectorRguez/wildchat-10k-filtered",
        "description": "WildChat 10k filtered dataset", 
        "output_dir": "data/wildchat-10k-filtered"
    },
    "wildchat-1k-dpo-annotation": {
        "repo": "HectorRguez/wildchat-1k-dpo-annotation",
        "description": "WildChat 1k dataset for DPO annotation",
        "output_dir": "data/wildchat-1k-dpo-annotation"
    }
}

def check_dataset_exists(repo_id: str) -> bool:
    """Check if dataset exists on Hugging Face"""
    try:
        api = HfApi()
        api.dataset_info(repo_id)
        return True
    except Exception as e:
        print(f"❌ Dataset {repo_id} not found: {e}")
        return False

def download_dataset(dataset_key: str, output_format: str = "json", cache_dir: Optional[str] = None) -> bool:
    """
    Download a specific dataset
    
    Args:
        dataset_key: Key from DATASETS dict
        output_format: Format to save ('json', 'parquet', 'disk')
        cache_dir: Cache directory for datasets library
    
    Returns:
        bool: Success status
    """
    if dataset_key not in DATASETS:
        print(f"❌ Unknown dataset: {dataset_key}")
        print(f"Available datasets: {list(DATASETS.keys())}")
        return False
    
    config = DATASETS[dataset_key]
    repo_id = config["repo"]
    output_dir = Path(config["output_dir"])
    
    print(f"📥 Downloading {config['description']}...")
    print(f"   Repository: {repo_id}")
    print(f"   Output: {output_dir}")
    
    # Check if dataset exists
    if not check_dataset_exists(repo_id):
        return False
    
    try:
        # Load dataset
        print("   Loading from Hugging Face...")
        dataset = load_dataset(repo_id, cache_dir=cache_dir)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save based on format
        if output_format == "json":
            # Save each split as JSON
            for split_name, split_data in dataset.items():
                output_file = output_dir / f"{split_name}.json"
                print(f"   Saving {split_name} split to {output_file}")
                
                # Convert to list of dicts and save
                data = split_data.to_list()
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
                print(f"   ✅ Saved {len(data)} examples to {output_file}")
        
        elif output_format == "parquet":
            # Save each split as Parquet
            for split_name, split_data in dataset.items():
                output_file = output_dir / f"{split_name}.parquet"
                print(f"   Saving {split_name} split to {output_file}")
                split_data.to_parquet(str(output_file))
                print(f"   ✅ Saved {len(split_data)} examples to {output_file}")
        
        elif output_format == "disk":
            # Save using datasets format
            print(f"   Saving to disk format in {output_dir}")
            dataset.save_to_disk(str(output_dir))
            print(f"   ✅ Saved dataset to {output_dir}")
        
        # Save metadata
        metadata = {
            "dataset_key": dataset_key,
            "repo_id": repo_id,
            "description": config["description"],
            "format": output_format,
            "splits": list(dataset.keys()),
            "total_examples": sum(len(split) for split in dataset.values())
        }
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"🎉 Successfully downloaded {config['description']}")
        print(f"   Total examples: {metadata['total_examples']}")
        print(f"   Splits: {metadata['splits']}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Error downloading {repo_id}: {e}")
        return False

def download_all_datasets(output_format: str = "json", cache_dir: Optional[str] = None) -> None:
    """Download all available datasets"""
    print("📦 Downloading all HectorRguez datasets...\n")
    
    success_count = 0
    total_count = len(DATASETS)
    
    for dataset_key in DATASETS.keys():
        if download_dataset(dataset_key, output_format, cache_dir):
            success_count += 1
    
    print(f"📊 Download Summary:")
    print(f"   Successful: {success_count}/{total_count}")
    print(f"   Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All datasets downloaded successfully!")
    else:
        print("⚠️  Some downloads failed. Check the logs above.")

def list_datasets() -> None:
    """List all available datasets"""
    print("📋 Available datasets:\n")
    
    for key, config in DATASETS.items():
        print(f"🔹 {key}")
        print(f"   Repository: {config['repo']}")
        print(f"   Description: {config['description']}")
        print(f"   Output: {config['output_dir']}")
        print()

def main():
    parser = argparse.ArgumentParser(
        description="Download HectorRguez datasets from Hugging Face",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py --list
  python download_datasets.py --all
  python download_datasets.py --dataset wildchat-1k-filtered-ads
  python download_datasets.py --dataset wildchat-1k-dpo-annotation
  python download_datasets.py --dataset wildchat-1k-filtered-ads --format parquet
  python download_datasets.py --all --cache-dir ./hf_cache
        """
    )
    
    parser.add_argument(
        "--dataset", 
        choices=list(DATASETS.keys()),
        help="Download specific dataset"
    )
    
    parser.add_argument(
        "--all", 
        action="store_true",
        help="Download all datasets"
    )
    
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List available datasets"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "parquet", "disk"],
        default="json",
        help="Output format (default: json)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory for datasets library"
    )
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        list_datasets()
        return
    
    # Handle download commands
    if args.all:
        download_all_datasets(args.format, args.cache_dir)
    elif args.dataset:
        download_dataset(args.dataset, args.format, args.cache_dir)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
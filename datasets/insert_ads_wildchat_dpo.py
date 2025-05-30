import json
import requests
import time
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import argparse
import os

class DPODatasetGenerator:
    def __init__(self, base_url: str = 'http://localhost:8888', headers: Dict = None):
        self.base_url = base_url
        self.headers = headers or {'Content-Type': 'application/json'}
        self.processed_count = 0
        self.failed_count = 0
        
        # Test server connection
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                print(f"✅ Connected to server at {base_url}")
            else:
                raise Exception(f"Server not healthy: {response.status_code}")
        except Exception as e:
            raise Exception(f"Cannot connect to server at {base_url}: {e}")

    def load_dataset(self, filepath: str) -> List[Dict]:
        """Load dataset from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)



    def generate_api_response(self, text: str, retries: int = 3) -> Dict[str, Any]:
        """Generate response using the API with retry logic."""
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/insert_native_ads",
                    headers=self.headers,
                    json={'text': text},
                    timeout=30  # 30 second timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    print(f"❌ Request failed after {retries} attempts: {e}")
        return None

    def generate_diverse_responses(self, item: Dict) -> tuple:
        """Generate two diverse responses for the same item using API calls"""
        
        original_answer = item.get('original_answer', '') or item.get('answer', '')
        
        if not original_answer:
            return None, None
        
        # Generate first response (baseline)
        result1 = self.generate_api_response(original_answer)
        
        # Generate second response (with different parameters handled by backend)
        result2 = self.generate_api_response(original_answer)
        
        response1 = None
        response2 = None
        
        if result1 and 'text_with_ads' in result1:
            response1 = {
                'answer': result1['text_with_ads'],
                'ads_inserted': True,
                'ad_metadata': {
                    'products_found': len(result1.get('related_products', [])),
                    'top_products': result1.get('related_products', [])[:3]
                } if 'related_products' in result1 else {}
            }
            self.processed_count += 1
        else:
            response1 = {
                'answer': original_answer,
                'ads_inserted': False,
                'enhancement_failed': True,
                'error_reason': 'api_failure_response1'
            }
            self.failed_count += 1
        
        if result2 and 'text_with_ads' in result2:
            response2 = {
                'answer': result2['text_with_ads'],
                'ads_inserted': True,
                'ad_metadata': {
                    'products_found': len(result2.get('related_products', [])),
                    'top_products': result2.get('related_products', [])[:3]
                } if 'related_products' in result2 else {}
            }
            self.processed_count += 1
        else:
            response2 = {
                'answer': original_answer,
                'ads_inserted': False,
                'enhancement_failed': True,
                'error_reason': 'api_failure_response2'
            }
            self.failed_count += 1
        
        return response1, response2

    def write_incremental_json(self, output_file: str, new_item: Dict[str, Any], is_first: bool = False):
        """Write JSON incrementally to file."""
        if is_first:
            # Start new file with array opening
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump([new_item], f, indent=2, ensure_ascii=False)
        else:
            # Read existing data, append new item, and write back
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
                
                existing_data.append(new_item)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_data, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"❌ Error writing incremental JSON: {e}")

    def get_resume_point(self, output_file: str) -> int:
        """Get the number of items already processed if resuming."""
        if not os.path.exists(output_file):
            return 0
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            return len(existing_data)
        except Exception as e:
            print(f"⚠️  Error reading existing file for resume: {e}")
            return 0

    def create_dpo_dataset(self, input_dataset_path: str, output_path: str, max_items: int = 10000, 
                          delay_between_requests: float = 1.0):
        """
        Create DPO dataset by generating diverse responses for each item via API
        """
        
        print("Loading input dataset...")
        input_dataset = self.load_dataset(input_dataset_path)
        
        print(f"Input dataset size: {len(input_dataset)}")
        
        # Limit to max_items if dataset is larger
        if len(input_dataset) > max_items:
            input_dataset = input_dataset[:max_items]
            print(f"⚠️  Dataset has {len(input_dataset)} items, limiting to first {max_items} for processing")
        
        expected_items = len(input_dataset)
        print(f"Processing {expected_items} items")
        
        # Check if we need to resume processing
        resume_from = self.get_resume_point(output_path)
        if resume_from > 0:
            print(f"🔄 Resuming from item {resume_from + 1}")
        
        # Reset counters for this session
        session_processed = 0
        session_failed = 0
        
        # Process items starting from resume point
        items_to_process = input_dataset[resume_from:]
        
        merged_pairs = []
        
        print(f"🔄 Generating diverse response pairs via API...")
        
        # Create progress bar
        pbar = tqdm(enumerate(items_to_process), 
                   total=len(items_to_process), 
                   desc="Processing",
                   initial=0)
        
        for idx, item in pbar:
            actual_idx = resume_from + idx
            
            # Generate two diverse responses using API
            response1, response2 = self.generate_diverse_responses(item)
            
            if response1 is None or response2 is None:
                print(f"❌ Failed to generate responses for item {actual_idx + 1}")
                session_failed += 1
                continue
            
            # Create DPO pair
            merged_pair = {
                "prompt": item.get('question', ''),  # Use the original question as prompt
                "response_1": response1.get('answer', ''),
                "response_2": response2.get('answer', ''),
                "preference": "",  # Empty field for future annotation
                "metadata": {
                    "question": item.get('question', ''),
                    "original_answer": item.get('original_answer', '') or item.get('answer', ''),
                    "response_1_ad_metadata": response1.get('ad_metadata', {}),
                    "response_2_ad_metadata": response2.get('ad_metadata', {}),
                    "response_1_ads_inserted": response1.get('ads_inserted', False),
                    "response_2_ads_inserted": response2.get('ads_inserted', False),
                    "processing_index": actual_idx,
                    "generation_method": "api_diverse_sampling"
                }
            }
            
            # Write item immediately
            is_first_item = (actual_idx == 0)
            self.write_incremental_json(output_path, merged_pair, is_first_item)
            
            session_processed += 1
            
            # Update progress bar
            pbar.set_postfix({
                'processed': session_processed,
                'failed': session_failed,
                'total': actual_idx + 1
            })
            
            # Add delay to avoid overwhelming server
            time.sleep(delay_between_requests)
        
        pbar.close()
        
        # Final verification
        final_count = self.get_resume_point(output_path)
        
        # Print summary
        print("\n" + "=" * 60)
        print("PROCESSING SUMMARY")
        print("=" * 60)
        print(f"Total items in input dataset: {len(input_dataset)}")
        print(f"Items processed limit: {max_items}")
        print(f"DPO pairs in output file: {final_count}")
        print(f"Session - Processed: {session_processed}")
        print(f"Session - Failed: {session_failed}")
        if len(items_to_process) > 0:
            success_rate = (session_processed / len(items_to_process) * 100)
            print(f"Session success rate: {success_rate:.1f}%")
        print(f"DPO dataset saved to: {output_path}")


def main():
    """Main function to run the DPO dataset generation."""
    parser = argparse.ArgumentParser(description='Generate DPO dataset using API for diverse responses')
    parser.add_argument('--input_file', type=str,
                       default='datasets/wildchat_10000_robust_sample.json',
                       help='Input JSON file with Q&A pairs that have ad metadata')
    parser.add_argument('--output_file', type=str,
                       default='datasets/wildchat_10000_robust_ads_dpo.json',
                       help='Output JSON file with DPO pairs')
    parser.add_argument('--server_url', type=str,
                       default='http://localhost:8888',
                       help='URL of the server with native ads API')
    parser.add_argument('--max_items', type=int,
                       default=10000,
                       help='Maximum number of items to process')
    parser.add_argument('--delay', type=float,
                       default=1.0,
                       help='Delay between requests in seconds')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration even if output file exists')
    
    args = parser.parse_args()
    
    print("🚀 DPO Dataset Generation Tool (API-based)")
    print("=" * 50)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"❌ Error: Input file '{args.input_file}' not found!")
        print("Please make sure your JSON file exists and the path is correct.")
        return
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    try:
        generator = DPODatasetGenerator(args.server_url)
    except Exception as e:
        print(f"❌ Failed to initialize generator: {e}")
        return
    
    # Override completion check if force flag is used
    if args.force and os.path.exists(args.output_file):
        print("🔄 Force mode enabled - removing existing output file")
        os.remove(args.output_file)
    
    # Load dataset for info
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        if not dataset:
            print("❌ Error: Dataset is empty!")
            return
        
        print(f"\n📊 Found {len(dataset)} Q&A pairs in the dataset.")
        print(f"📁 Input file: {args.input_file}")
        print(f"📁 Output file: {args.output_file}")
        print(f"🔢 Max items to process: {min(args.max_items, len(dataset))}")
        if len(dataset) > args.max_items:
            print(f"⚠️  Note: Dataset will be limited to first {args.max_items} items")
        
        # Ask user if they want to proceed
        if not args.force:
            proceed = input(f"\nProceed with generating DPO pairs for up to {min(args.max_items, len(dataset))} items? (y/n): ").lower().strip()
            if proceed != 'y':
                print("Operation cancelled.")
                return
        
        # Process the dataset
        print(f"\n🔄 Starting DPO dataset generation...")
        print(f"💾 Files will be written incrementally (item by item)")
        print(f"⏭️  Incomplete files will be resumed automatically")
        print(f"🎯 Each item will generate 2 diverse responses via API")
        
        generator.create_dpo_dataset(
            input_dataset_path=args.input_file,
            output_path=args.output_file,
            max_items=args.max_items,
            delay_between_requests=args.delay
        )
        
        print(f"\n🎉 DPO dataset generation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")


if __name__ == "__main__":
    main()
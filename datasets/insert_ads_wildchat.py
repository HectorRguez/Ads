import requests
import json
import time
from textwrap import fill
import os
from typing import List, Dict, Any
from tqdm import tqdm
import argparse

# Server configuration
BASE_URL = 'http://localhost:8888'
HEADERS = {'Content-Type': 'application/json'}

class DatasetEnhancer:
    def __init__(self, base_url: str = BASE_URL, headers: Dict = HEADERS):
        self.base_url = base_url
        self.headers = headers
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
        
    def print_boxed_text(self, title: str, content: str, width: int = 78) -> str:
        """Helper function to print text in a box."""
        border = "+" + "-" * width + "+"
        title_line = f"|{title.center(width)}|"
        
        lines = [border, title_line, border]
        
        # Wrap content
        wrapped_lines = fill(content, width=width-4).split('\n')
        
        for line in wrapped_lines:
            padded_line = f"| {line.ljust(width-2)} |"
            lines.append(padded_line)
        
        lines.append(border)
        return '\n'.join(lines)
    
    def is_output_complete(self, output_file: str, expected_items: int) -> bool:
        """Check if output file exists and is complete."""
        if not os.path.exists(output_file):
            return False
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            
            if len(existing_data) == expected_items:
                print(f"✅ Output file already complete: {output_file} ({len(existing_data)} items)")
                return True
            else:
                print(f"⚠️  Incomplete output file found: {output_file} ({len(existing_data)}/{expected_items} items)")
                return False
        except Exception as e:
            print(f"⚠️  Error reading existing output file {output_file}: {e}")
            return False
        
        return False
    
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
    
    def insert_native_ads(self, text: str, prompt: str = "", retries: int = 3) -> Dict[str, Any]:
        """Insert native ads into text using the API with retry logic."""
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.base_url}/insert_native_ads",
                    headers=self.headers,
                    json={'text': text, 'prompt': prompt},
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
    
    def enhance_qa_pair(self, qa_item: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a single Q&A pair with native advertisements."""
        original_answer = qa_item.get('answer', '')
        question = qa_item.get('question', '')
        
        if not original_answer:
            self.failed_count += 1
            failed_item = qa_item.copy()
            failed_item['ads_inserted'] = False
            failed_item['enhancement_failed'] = True
            failed_item['error_reason'] = 'empty_answer'
            return failed_item
        
        # Call the native ads API with both prompt (question) and text (answer)
        result = self.insert_native_ads(original_answer, prompt=question)
        
        if result and 'text_with_ads' in result:
            enhanced_item = qa_item.copy()
            enhanced_item['original_answer'] = original_answer
            enhanced_item['answer'] = result['text_with_ads']
            enhanced_item['ads_inserted'] = True
            
            # Add metadata about the ads
            if 'related_products' in result:
                enhanced_item['ad_metadata'] = {
                    'products_found': len(result['related_products']),
                    'top_products': result['related_products'][:3] if result['related_products'] else []
                }
            
            self.processed_count += 1
            return enhanced_item
        else:
            self.failed_count += 1
            # Return original item with failure flag
            failed_item = qa_item.copy()
            failed_item['ads_inserted'] = False
            failed_item['enhancement_failed'] = True
            failed_item['error_reason'] = 'api_failure'
            return failed_item
    
    def process_dataset(self, input_file: str, output_file: str, max_items: int = 1000, 
                       delay_between_requests: float = 0.1) -> List[Dict[str, Any]]:
        """Process the dataset and create enhanced version with incremental writing."""
        try:
            # Load the dataset
            print(f"📁 Loading dataset from {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            original_count = len(dataset)
            print(f"Loaded {original_count} Q&A pairs")
            
            # Limit to max_items if dataset is larger
            if len(dataset) > max_items:
                dataset = dataset[:max_items]
                print(f"⚠️  Dataset has {original_count} items, limiting to first {max_items} for processing")
            
            expected_items = len(dataset)
            print(f"Processing {expected_items} items")
            
            # Check if output is already complete
            if self.is_output_complete(output_file, expected_items):
                return []
            
            # Check if we need to resume processing
            resume_from = self.get_resume_point(output_file)
            if resume_from > 0:
                print(f"🔄 Resuming from item {resume_from + 1}")
            
            # Reset counters for this session
            session_processed = 0
            session_failed = 0
            
            # Process items starting from resume point
            items_to_process = dataset[resume_from:]
            
            print(f"🔄 Enhancing Q&A pairs with native ads...")
            
            # Create progress bar
            pbar = tqdm(enumerate(items_to_process), 
                       total=len(items_to_process), 
                       desc="Enhancing",
                       initial=0)
            
            for idx, qa_item in pbar:
                actual_idx = resume_from + idx
                
                # Enhance the item
                enhanced_item = self.enhance_qa_pair(qa_item)
                
                # Track session stats
                if enhanced_item.get('ads_inserted', False):
                    session_processed += 1
                else:
                    session_failed += 1
                
                # Add processing metadata
                enhanced_item['processing_index'] = actual_idx
                enhanced_item['original_dataset_size'] = original_count
                enhanced_item['max_items_processed'] = max_items
                
                # Write item immediately
                is_first_item = (actual_idx == 0)
                self.write_incremental_json(output_file, enhanced_item, is_first_item)
                
                # Update progress bar
                pbar.set_postfix({
                    'enhanced': session_processed,
                    'failed': session_failed,
                    'total_processed': actual_idx + 1
                })
                
                # Add delay to avoid overwhelming server
                time.sleep(delay_between_requests)
            
            pbar.close()
            
            # Final verification
            final_count = self.get_resume_point(output_file)
            
            # Print summary
            print("\n" + "=" * 60)
            print("PROCESSING SUMMARY")
            print("=" * 60)
            print(f"Total items in dataset: {original_count}")
            print(f"Items processed limit: {max_items}")
            print(f"Items in output file: {final_count}")
            print(f"Session - Enhanced: {session_processed}")
            print(f"Session - Failed: {session_failed}")
            if final_count > 0:
                success_rate = (session_processed / len(items_to_process) * 100) if items_to_process else 0
                print(f"Session success rate: {success_rate:.1f}%")
            print(f"Enhanced dataset saved to: {output_file}")
            
            # Load and return final dataset
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
            
        except FileNotFoundError:
            print(f"❌ Error: Input file {input_file} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Error: Invalid JSON in input file: {e}")
            return []
        except Exception as e:
            print(f"❌ Error: Unexpected error: {e}")
            return []
    
    def preview_enhancement(self, qa_item: Dict[str, Any]) -> None:
        """Preview how a single Q&A pair would be enhanced."""
        print("\n" + "="*80)
        print("ENHANCEMENT PREVIEW")
        print("="*80)
        
        print(self.print_boxed_text("QUESTION", qa_item.get('question', 'N/A')))
        print()
        print(self.print_boxed_text("ORIGINAL ANSWER", qa_item.get('answer', 'N/A')))
        
        enhanced_item = self.enhance_qa_pair(qa_item)
        
        if enhanced_item.get('ads_inserted'):
            print()
            print(self.print_boxed_text("ANSWER WITH ADS", enhanced_item.get('answer', 'N/A')))
            
            if 'ad_metadata' in enhanced_item:
                print(f"\n📊 Ad Metadata:")
                print(f"  Products found: {enhanced_item['ad_metadata']['products_found']}")
                if enhanced_item['ad_metadata']['top_products']:
                    print(f"  Top products:")
                    for prod in enhanced_item['ad_metadata']['top_products']:
                        print(f"    - {prod.get('name', 'Unknown')} (similarity: {prod.get('similarity', 0):.3f})")
        else:
            print("\n❌ Enhancement failed!")
            if 'error_reason' in enhanced_item:
                print(f"Reason: {enhanced_item['error_reason']}")


def main():
    """Main function to run the dataset enhancement."""
    parser = argparse.ArgumentParser(description='Enhance Q&A dataset with native advertisements')
    parser.add_argument('--input_file', type=str,
                       default='datasets/wildchat_10k_filtered.json',
                       help='Input JSON file with Q&A pairs')
    parser.add_argument('--output_file', type=str,
                       default='datasets/wildchat_1k_filtered_ads.json',
                       help='Output JSON file with enhanced Q&A pairs')
    parser.add_argument('--server_url', type=str,
                       default='http://localhost:8888',
                       help='URL of the server with native ads API')
    parser.add_argument('--max_items', type=int,
                       default=1000,
                       help='Maximum number of items to process')
    parser.add_argument('--delay', type=float,
                       default=0.1,
                       help='Delay between requests in seconds')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration even if output file exists')
    parser.add_argument('--preview_only', action='store_true',
                       help='Only show preview of first item, do not process dataset')
    
    args = parser.parse_args()
    
    print("🚀 Native Ads Dataset Enhancement Tool")
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
    
    # Initialize enhancer
    try:
        enhancer = DatasetEnhancer(args.server_url)
    except Exception as e:
        print(f"❌ Failed to initialize enhancer: {e}")
        return
    
    # Override completion check if force flag is used
    if args.force:
        print("🔄 Force mode enabled - will regenerate file")
        enhancer.is_output_complete = lambda *args: False
    
    # Load dataset for preview and info
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
        
        # Preview mode
        if args.preview_only:
            enhancer.preview_enhancement(dataset[0])
            return
        
        # Interactive mode if no explicit flags
        if not args.force:
            preview = input("\nWould you like to preview enhancement with the first item? (y/n): ").lower().strip()
            if preview == 'y':
                enhancer.preview_enhancement(dataset[0])
        
            # Ask user if they want to proceed
            proceed = input(f"\nProceed with enhancing up to {min(args.max_items, len(dataset))} items? (y/n): ").lower().strip()
            if proceed != 'y':
                print("Operation cancelled.")
                return
        
        # Process the dataset
        print(f"\n🔄 Starting enhancement process...")
        print(f"💾 Files will be written incrementally (item by item)")
        if not args.force:
            print(f"⏭️  Incomplete files will be resumed automatically")
        
        enhanced_dataset = enhancer.process_dataset(
            input_file=args.input_file,
            output_file=args.output_file,
            max_items=args.max_items,
            delay_between_requests=args.delay
        )
        
        if enhanced_dataset:
            print(f"\n🎉 Enhancement completed successfully!")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")


if __name__ == "__main__":
    main()
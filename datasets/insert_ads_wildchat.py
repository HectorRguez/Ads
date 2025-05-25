import requests
import json
import time
from textwrap import fill
import os
from typing import List, Dict, Any
from tqdm import tqdm
import random

# Server configuration
BASE_URL = 'http://localhost:8888'
HEADERS = {'Content-Type': 'application/json'}

class DatasetEnhancer:
    def __init__(self, base_url: str = BASE_URL, headers: Dict = HEADERS):
        self.base_url = base_url
        self.headers = headers
        self.processed_count = 0
        self.failed_count = 0
        
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
    
    def insert_native_ads(self, text: str, retries: int = 3) -> Dict[str, Any]:
        """Insert native ads into text using the API with retry logic."""
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
                else:
                    if attempt < retries - 1:
                        time.sleep(2 ** attempt)  # Exponential backoff
                    
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
        return None
    
    def enhance_qa_pair(self, qa_item: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance a single Q&A pair with native advertisements."""
        original_answer = qa_item.get('answer', '')
        
        if not original_answer:
            return qa_item
        
        # Call the native ads API
        result = self.insert_native_ads(original_answer)
        
        if result and 'modified_text' in result:
            enhanced_item = qa_item.copy()
            enhanced_item['original_answer'] = original_answer
            enhanced_item['answer'] = result['modified_text']
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
            return failed_item
    
    def process_dataset(self, input_file: str, output_file: str, batch_size: int = 10, 
                       delay_between_batches: float = 1.0) -> List[Dict[str, Any]]:
        """Process the entire dataset and create enhanced version."""
        try:
            # Load the dataset
            print(f"Loading dataset from {input_file}")
            with open(input_file, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            print(f"Loaded {len(dataset)} Q&A pairs")
            
            enhanced_dataset = []
            
            # Process with progress bar
            with tqdm(total=1000, desc="Processing Q&A pairs", unit="item") as pbar:
                for i in range(1000):
                    enhanced_item = self.enhance_qa_pair(dataset[i])
                    enhanced_dataset.append(enhanced_item)
                    pbar.update(1)
            
            # Save the enhanced dataset
            print(f"\nSaving enhanced dataset to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_dataset, f, indent=2, ensure_ascii=False)
            
            # Print summary
            print("\n" + "=" * 60)
            print("PROCESSING SUMMARY")
            print("=" * 60)
            print(f"Total items processed: {len(dataset)}")
            print(f"Successfully enhanced: {self.processed_count}")
            print(f"Failed to enhance: {self.failed_count}")
            print(f"Success rate: {(self.processed_count/len(dataset)*100):.1f}%")
            print(f"Enhanced dataset saved to: {output_file}")
            
            return enhanced_dataset
            
        except FileNotFoundError:
            print(f"Error: Input file {input_file} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in input file: {e}")
            return []
        except Exception as e:
            print(f"Error: Unexpected error: {e}")
            return []
    
    def preview_enhancement(self, qa_item: Dict[str, Any]) -> None:
        """Preview how a single Q&A pair would be enhanced."""
        print("\n" + "="*80)
        print("ENHANCEMENT PREVIEW")
        print("="*80)
        
        print(self.print_boxed_text("QUESTION", qa_item.get('question', 'N/A')))
        print()
        print(self.print_boxed_text("ANSWER", qa_item.get('answer', 'N/A')))
        
        enhanced_item = self.enhance_qa_pair(qa_item)
        
        if enhanced_item.get('ads_inserted'):
            print()
            print(self.print_boxed_text("ANSWER WITH ADS", enhanced_item.get('answer', 'N/A')))
            
            if 'ad_metadata' in enhanced_item:
                print(f"\nAd Metadata:")
                print(f"  Products found: {enhanced_item['ad_metadata']['products_found']}")
                if enhanced_item['ad_metadata']['top_products']:
                    print(f"  Top products:")
                    for prod in enhanced_item['ad_metadata']['top_products']:
                        print(f"    - {prod.get('name', 'Unknown')} (similarity: {prod.get('similarity', 0):.3f})")
        else:
            print("\nEnhancement failed!")


def main():
    """Main function to run the dataset enhancement."""
    enhancer = DatasetEnhancer()
    
    # Configuration
    input_file = "datasets/wildchat_10k_filtered.json"       # Your input JSON file
    output_file = "datasets/wildchat_1k_filtered_ads.json"   # Output file with ads
    
    print("Native Ads Dataset Enhancement Tool")
    print("=" * 50)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print("Please make sure your JSON file is in the same directory and named correctly.")
        return
    
    # Preview mode - test with first item
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        if dataset:
            print(f"\nFound {len(dataset)} Q&A pairs in the dataset.")
            
            # Ask user if they want to preview first
            preview = input("\nWould you like to preview enhancement with the first item? (y/n): ").lower().strip()
            if preview == 'y':
                enhancer.preview_enhancement(dataset[0])
            
            # Ask user if they want to proceed with full processing
            proceed = input(f"\nProceed with enhancing all {len(dataset)} items? (y/n): ").lower().strip()
            if proceed == 'y':
                # Process the entire dataset
                enhancer.process_dataset(
                    input_file=input_file,
                    output_file=output_file,
                )
            
    except Exception as e:
        print(f"Error loading dataset: {e}")


if __name__ == "__main__":
    main()
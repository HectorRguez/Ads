#!/usr/bin/env python3
"""
Insert native ads into TRUE baseline responses
This creates CSV files with baseline responses enhanced with ads for evaluation
"""

import os
import pandas as pd
import requests
import time
from tqdm import tqdm
import argparse
import csv

class TRUEBaselineAdEnhancer:
    def __init__(self, server_url="http://localhost:8888"):
        self.server_url = server_url
        self.headers = {'Content-Type': 'application/json'}
        
        # Test server connection
        try:
            response = requests.get(f"{server_url}/health")
            if response.status_code == 200:
                print(f"Connected to server at {server_url}")
            else:
                raise Exception(f"Server not healthy: {response.status_code}")
        except Exception as e:
            raise Exception(f"Cannot connect to server at {server_url}: {e}")
    
    def insert_native_ads(self, text: str, prompt: str = "", retries: int = 3):
        """Insert native ads into text using the API with retry logic."""
        for attempt in range(retries):
            try:
                response = requests.post(
                    f"{self.server_url}/insert_native_ads",
                    headers=self.headers,
                    json={'text': text, 'prompt': prompt},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get('text_with_ads', text)  # Return enhanced text or original
                elif response.status_code == 429:  # Rate limited
                    wait_time = 2 ** attempt
                    print(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    if attempt < retries - 1:
                        print(f"Server error {response.status_code}, retrying in {2**attempt}s...")
                        time.sleep(2 ** attempt)
                        continue
                    else:
                        print(f"Server error {response.status_code}: {response.text}")
                        return None
                        
            except requests.exceptions.RequestException as e:
                if attempt < retries - 1:
                    print(f"Request error, retrying in {2**attempt}s: {e}")
                    time.sleep(2 ** attempt)
                    continue
                else:
                    print(f"Request failed after {retries} attempts: {e}")
                    return None
        
        return None
    
    def is_output_complete(self, output_file, expected_rows):
        """Check if output file exists and is complete."""
        if not os.path.exists(output_file):
            return False
        
        try:
            existing_df = pd.read_csv(output_file)
            if len(existing_df) == expected_rows:
                print(f"Output file already complete: {output_file} ({len(existing_df)} rows)")
                return True
            else:
                print(f"Incomplete output file found: {output_file} ({len(existing_df)}/{expected_rows} rows)")
                return False
        except Exception as e:
            print(f"Error reading existing output file {output_file}: {e}")
            return False
        
        return False
    
    def get_resume_point(self, output_file):
        """Get the number of rows already processed if resuming."""
        if not os.path.exists(output_file):
            return 0
        
        try:
            existing_df = pd.read_csv(output_file)
            return len(existing_df)
        except Exception as e:
            print(f"Error reading existing file for resume: {e}")
            return 0
    
    def write_csv_header(self, output_file):
        """Write CSV header to output file."""
        headers = [
            'grounding',
            'generated_text',
            'generated_text_with_ads', 
            'label',
            'dataset',
            'original_dataset_size',
            'processed_rows',
            'ads_inserted_successfully'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def append_csv_row(self, output_file, row_data):
        """Append a single row to the CSV file."""
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
    
    def process_true_dataset(self, input_csv, output_dir, dataset_name):
        """
        Process a TRUE dataset and enhance baseline responses with ads.
        
        TRUE datasets have 'grounding' and 'generated_text' (baseline responses).
        We'll enhance the 'generated_text' with native ads.
        """
        print(f"\nProcessing {dataset_name} dataset...")
        
        # Load TRUE dataset
        df = pd.read_csv(input_csv)
        original_count = len(df)
        print(f"Loaded {original_count} examples")
        
        # Limit to first 1000 rows if dataset is larger
        if len(df) > 1000:
            df = df.head(1000)
            print(f"Dataset has {original_count} rows, limiting to first 1000 for processing")
        
        expected_rows = len(df)
        print(f"Processing {expected_rows} examples")
        
        # Check required columns
        required_cols = ['grounding', 'generated_text', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return None
        
        # Define output file
        output_file = os.path.join(output_dir, f"{dataset_name.lower()}_baseline_with_ads.csv")
        
        # Check if output is already complete
        if self.is_output_complete(output_file, expected_rows):
            return output_file
        
        # Check if we need to resume processing
        resume_from = self.get_resume_point(output_file)
        if resume_from > 0:
            print(f"Resuming from row {resume_from + 1}")
        else:
            # Write header for new file
            self.write_csv_header(output_file)
        
        # Process rows starting from resume point
        failed_count = 0
        success_count = 0
        
        print(f"Enhancing baseline responses with native ads...")
        
        # Create progress bar starting from resume point
        rows_to_process = df.iloc[resume_from:]
        pbar = tqdm(rows_to_process.iterrows(), 
                   total=len(rows_to_process), 
                   desc="Enhancing",
                   initial=0)
        
        for idx, row in pbar:
            # Get the baseline generated text and grounding for context
            grounding = str(row['grounding'])
            generated_text = str(row['generated_text'])
            label = row['label']
            
            # Skip if generated text is empty or too short
            if len(generated_text.strip()) < 10:
                enhanced_text = generated_text  # Keep original
                ads_success = False
                failed_count += 1
            else:
                # Create a simple prompt from the grounding for context
                prompt = self._create_context_prompt(grounding, dataset_name)
                
                # Enhance the baseline response with ads
                enhanced_text = self.insert_native_ads(generated_text, prompt=prompt)
                
                if enhanced_text and enhanced_text != generated_text:
                    ads_success = True
                    success_count += 1
                else:
                    enhanced_text = generated_text  # Keep original if enhancement failed
                    ads_success = False
                    failed_count += 1
            
            # Prepare row data
            row_data = [
                grounding,
                generated_text,
                enhanced_text,
                label,
                dataset_name,
                original_count,
                expected_rows,
                ads_success
            ]
            
            # Write row immediately
            self.append_csv_row(output_file, row_data)
            
            # Update progress bar with current status
            pbar.set_postfix({
                'enhanced': success_count,
                'failed': failed_count,
                'completed': (idx - resume_from + 1)
            })
            
            # Add small delay to avoid overwhelming server
            time.sleep(0.1)
        
        pbar.close()
        
        # Final verification
        final_rows = self.get_resume_point(output_file)
        
        print(f"Saved {final_rows} enhanced responses to: {output_file}")
        print(f"   Successfully enhanced: {success_count}")
        print(f"   Failed enhancements: {failed_count}")
        if original_count > 1000:
            print(f"   Note: Original dataset had {original_count} rows, processed first 1000")
        
        return output_file
    
    def _create_context_prompt(self, grounding, dataset_name):
        """
        Create a simple context prompt from grounding text for better ad targeting.
        """
        # Truncate grounding to reasonable length for prompt
        max_prompt_length = 200
        truncated_grounding = grounding[:max_prompt_length]
        if len(grounding) > max_prompt_length:
            truncated_grounding += "..."
        
        dataset_lower = dataset_name.lower()
        
        # Create dataset-appropriate context prompts
        if 'summeval' in dataset_lower or 'qags' in dataset_lower:
            return f"Context: News article summary - {truncated_grounding}"
        elif 'fever' in dataset_lower or 'vitc' in dataset_lower:
            return f"Context: Fact verification - {truncated_grounding}"
        elif 'dialfact' in dataset_lower:
            return f"Context: Dialogue response - {truncated_grounding}"
        elif 'frank' in dataset_lower:
            return f"Context: Text rewriting - {truncated_grounding}"
        else:
            return f"Context: {truncated_grounding}"

def main():
    parser = argparse.ArgumentParser(description='Enhance TRUE baseline responses with native ads')
    parser.add_argument('--data_dir', type=str,
                       default='/data/hector/datasets/true_datasets',
                       help='Directory containing TRUE datasets')
    parser.add_argument('--output_dir', type=str,
                       default='/data/hector/baseline_with_ads',
                       help='Directory to save enhanced baseline responses')
    parser.add_argument('--server_url', type=str,
                       default='http://localhost:8888',
                       help='URL of your ad insertion server')
    parser.add_argument('--datasets', nargs='+',
                       help='Specific datasets to process (leave empty for all)')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration even if output files exist')
    parser.add_argument('--delay', type=float,
                       default=0.1,
                       help='Delay between requests in seconds')
    parser.add_argument('--retries', type=int,
                       default=3,
                       help='Number of retries for failed requests')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize enhancer
    enhancer = TRUEBaselineAdEnhancer(args.server_url)
    
    # Override completion check if force flag is used
    if args.force:
        print("Force mode enabled - will regenerate all files")
        enhancer.is_output_complete = lambda *args: False
    
    # Define datasets to process (matching your actual files)
    all_datasets = [
        ('begin_dev_download.csv', 'BEGIN'),
        ('dialfact_valid_download.csv', 'DialFact'),
        ('frank_valid_download.csv', 'FRANK'),
        ('mnbm_download.csv', 'MNBM'),
        ('q2_download.csv', 'Q2'),
        ('qags_cnndm_download.csv', 'QAGS_CNNDM'),
        ('qags_xsum_download.csv', 'QAGS_XSum'),
        ('summeval_download.csv', 'SummEval'),
        ('vitc_dev_download.csv', 'VitaminC'),
        ('paws_download.csv', 'PAWS'),
        ('fever_dev_download.csv', 'FEVER'),
    ]
    
    # Filter datasets if specified
    if args.datasets:
        all_datasets = [(csv, name) for csv, name in all_datasets 
                       if name.lower() in [d.lower() for d in args.datasets]]
    
    print(f"Starting baseline response enhancement for {len(all_datasets)} datasets")
    print(f"Output directory: {args.output_dir}")
    print(f"Note: Datasets with >1000 rows will be limited to first 1000")
    print(f"Files will be written incrementally (line by line)")
    print(f"Retry attempts: {args.retries}, Delay: {args.delay}s")
    if not args.force:
        print(f"Complete files will be skipped automatically")
    
    enhanced_files = []
    
    # Process each dataset
    for csv_file, dataset_name in all_datasets:
        input_path = os.path.join(args.data_dir, csv_file)
        
        if not os.path.exists(input_path):
            print(f"Skipping {dataset_name}: file not found at {input_path}")
            continue
        
        try:
            output_file = enhancer.process_true_dataset(
                input_path, args.output_dir, dataset_name
            )
            if output_file:
                enhanced_files.append(output_file)
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
    
    print(f"\nBaseline enhancement completed!")
    print(f"Enhanced {len(enhanced_files)} files:")
    for file in enhanced_files:
        print(f"   - {file}")
    
    print(f"\nNext steps:")
    print(f"1. Use the TRUE model evaluator to assess factual consistency")
    print(f"2. Compare baseline vs baseline+ads performance")
    print(f"3. Compare with your Mistral model results")

if __name__ == "__main__":
    main()
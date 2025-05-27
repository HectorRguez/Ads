#!/usr/bin/env python3
"""
Stage 1: Generate responses from your model for TRUE benchmark datasets
This creates CSV files with your model's responses that can be evaluated later
"""

import os
import pandas as pd
import requests
import time
from tqdm import tqdm
import argparse
import csv

class TRUEResponseGenerator:
    def __init__(self, server_url="http://localhost:8888"):
        self.server_url = server_url
        self.headers = {'Content-Type': 'application/json'}
        
        # Test server connection
        try:
            response = requests.get(f"{server_url}/health")
            if response.status_code == 200:
                print(f"✅ Connected to server at {server_url}")
            else:
                raise Exception(f"Server not healthy: {response.status_code}")
        except Exception as e:
            raise Exception(f"Cannot connect to server at {server_url}: {e}")
    
    def generate_response(self, question, with_ads=False):
        """Generate response from your model."""
        endpoint = "/infer_local_native_ads" if with_ads else "/infer_local"
        
        data = {
            'question': question
        }
        
        try:
            response = requests.post(
                f"{self.server_url}{endpoint}", 
                headers=self.headers, 
                json=data,
                timeout=30  # 30 second timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                if with_ads:
                    return result.get('answer_with_ads', result.get('inferred', ''))
                else:
                    return result.get('inferred', result.get('answer', ''))
            else:
                print(f"❌ Server error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Request error: {e}")
            return None
    
    def is_output_complete(self, output_file, expected_rows):
        """Check if output file exists and is complete."""
        if not os.path.exists(output_file):
            return False
        
        try:
            existing_df = pd.read_csv(output_file)
            if len(existing_df) == expected_rows:
                # Check if all model responses are non-empty (excluding failed generations)
                non_empty_responses = existing_df['model_response'].fillna('').astype(str).str.strip()
                if len(non_empty_responses) > 0:  # At least some responses exist
                    print(f"✅ Output file already complete: {output_file} ({len(existing_df)} rows)")
                    return True
            else:
                print(f"⚠️  Incomplete output file found: {output_file} ({len(existing_df)}/{expected_rows} rows)")
                return False
        except Exception as e:
            print(f"⚠️  Error reading existing output file {output_file}: {e}")
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
            print(f"⚠️  Error reading existing file for resume: {e}")
            return 0
    
    def write_csv_header(self, output_file, with_ads):
        """Write CSV header to output file."""
        headers = [
            'original_grounding',
            'original_generated_text', 
            'original_label',
            'question_for_model',
            'model_response',
            'dataset',
            'with_ads',
            'original_dataset_size',
            'processed_rows'
        ]
        
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def append_csv_row(self, output_file, row_data):
        """Append a single row to the CSV file."""
        with open(output_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(row_data)
    
    def process_true_dataset(self, input_csv, output_dir, dataset_name, with_ads=False):
        """
        Process a TRUE dataset and generate responses.
        
        TRUE datasets have 'grounding' text, but we need to convert this to questions
        for your Q&A model. We'll use the 'generated_text' as a template for question generation.
        """
        print(f"\n🔄 Processing {dataset_name} dataset...")
        
        # Load TRUE dataset
        df = pd.read_csv(input_csv)
        original_count = len(df)
        print(f"Loaded {original_count} examples")
        
        # Limit to first 1000 rows if dataset is larger
        if len(df) > 1000:
            df = df.head(1000)
            print(f"⚠️  Dataset has {original_count} rows, limiting to first 1000 for processing")
        
        expected_rows = len(df)
        print(f"Processing {expected_rows} examples")
        
        # Check required columns
        required_cols = ['grounding', 'generated_text', 'label']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ Missing columns: {missing_cols}")
            return None
        
        # Define output file
        suffix = "_with_ads" if with_ads else "_no_ads"
        output_file = os.path.join(output_dir, f"{dataset_name.lower()}_responses{suffix}.csv")
        
        # Check if output is already complete
        if self.is_output_complete(output_file, expected_rows):
            return output_file
        
        # Check if we need to resume processing
        resume_from = self.get_resume_point(output_file)
        if resume_from > 0:
            print(f"🔄 Resuming from row {resume_from + 1}")
        else:
            # Write header for new file
            self.write_csv_header(output_file, with_ads)
        
        # Process rows starting from resume point
        failed_count = 0
        
        print(f"🔄 Generating responses ({'with ads' if with_ads else 'without ads'})...")
        
        # Create progress bar starting from resume point
        rows_to_process = df.iloc[resume_from:]
        pbar = tqdm(rows_to_process.iterrows(), 
                   total=len(rows_to_process), 
                   desc="Generating",
                   initial=0)
        
        for idx, row in pbar:
            # Create appropriate question for this dataset
            question = self._create_question(row['grounding'], row['generated_text'], dataset_name)
            
            # Generate response
            response = self.generate_response(question, with_ads=with_ads)
            
            if response is None:
                response = ""  # Empty response on failure
                failed_count += 1
            
            # Prepare row data
            row_data = [
                row['grounding'],
                row['generated_text'],
                row['label'],
                question,
                response,
                dataset_name,
                with_ads,
                original_count,
                expected_rows
            ]
            
            # Write row immediately
            self.append_csv_row(output_file, row_data)
            
            # Update progress bar with current status
            pbar.set_postfix({
                'failed': failed_count,
                'completed': (idx - resume_from + 1)
            })
            
            # Add small delay to avoid overwhelming server
            time.sleep(0.1)
        
        pbar.close()
        
        # Final verification
        final_rows = self.get_resume_point(output_file)
        
        print(f"✅ Saved {final_rows} responses to: {output_file}")
        print(f"   Failed generations: {failed_count}")
        if original_count > 1000:
            print(f"   Note: Original dataset had {original_count} rows, processed first 1000")
        
        return output_file
    
    def _create_question(self, grounding, generated_text, dataset_name):
        """
        Create dataset-appropriate questions based on the TRUE benchmark methodology.
        
        Each dataset has a specific task that the original text generation model was 
        performing. We replicate that task to generate comparable text from your model.
        """
        # Truncate grounding text to avoid token limits
        max_context_length = 800
        truncated_grounding = grounding[:max_context_length]
        if len(grounding) > max_context_length:
            truncated_grounding += "..."
        
        dataset_lower = dataset_name.lower()
        
        # SUMMARIZATION DATASETS
        if 'summeval' in dataset_lower:
            # SummEval: CNN/DailyMail article summarization
            # Original task: Generate a summary of a news article
            return f"Please write a concise summary of the following news article:\n\n{truncated_grounding}"
            
        elif 'qags_cnndm' in dataset_lower:
            # QAGS CNN/DM: Question-answering based summarization evaluation
            # Original task: Generate answers based on article content
            return f"Based on the following news article, please provide the key information and main points:\n\n{truncated_grounding}"
            
        elif 'qags_xsum' in dataset_lower:
            # QAGS XSum: Extreme summarization (BBC articles)
            # Original task: Generate one-sentence summaries
            return f"Please provide a one-sentence summary that captures the essence of this BBC article:\n\n{truncated_grounding}"
        
        # PARAPHRASE AND SEMANTIC SIMILARITY
        elif 'paws' in dataset_lower:
            # PAWS: Paraphrase Adversaries from Word Scrambling
            # Original task: Determine if two sentences have the same meaning
            # We ask for paraphrasing to test semantic consistency
            return f"Please rephrase the following sentence while preserving its exact meaning:\n\n{truncated_grounding}"
        
        # FACT VERIFICATION
        elif 'fever' in dataset_lower:
            # FEVER: Fact Extraction and VERification
            # Original task: Given evidence, classify claims as SUPPORTED/REFUTED/NOT ENOUGH INFO
            return f"Based on the following evidence, please explain what conclusions can be drawn and provide a factual analysis:\n\nEvidence: {truncated_grounding}"
        
        # DIALOGUE SYSTEMS
        elif 'dialfact' in dataset_lower:
            # DialFact: Dialogue factual consistency
            # Original task: Generate responses in dialogue that are factually consistent
            return f"Please continue this conversation with a factually accurate and helpful response:\n\n{truncated_grounding}"
        
        # QUESTION ANSWERING
        elif 'q2' in dataset_lower or 'q²' in dataset_lower:
            # Q²: Question generation and answering
            # Original task: Generate questions and answers from context
            return f"Based on the following context, please provide relevant questions and their answers:\n\n{truncated_grounding}"
        
        # TEXT SIMPLIFICATION AND REWRITING
        elif 'frank' in dataset_lower:
            # FRANK: Factual consistency in text rewriting
            # Original task: Rewrite text while maintaining factual accuracy
            return f"Please rewrite the following text to make it clearer and more accessible while maintaining all factual information:\n\n{truncated_grounding}"
        
        # OPEN-DOMAIN TEXT GENERATION
        elif 'begin' in dataset_lower:
            # BEGIN: Benchmark for text generation evaluation
            # Original task: Generate text continuations
            return f"Please continue or expand on the following text in a natural and informative way:\n\n{truncated_grounding}"
        
        # MULTI-REFERENCE EVALUATION
        elif 'mnbm' in dataset_lower:
            # MNBM: Multi-reference benchmark
            # Original task: Generate text with multiple valid references
            return f"Please provide an informative response based on the following context:\n\n{truncated_grounding}"
        
        # CLAIM VERIFICATION
        elif 'vitc' in dataset_lower or 'vitamin' in dataset_lower:
            # VitaminC: Fact verification with evidence
            # Original task: Verify claims against evidence passages
            return f"Based on the following evidence, please verify and explain whether the associated claims are factually supported:\n\nEvidence: {truncated_grounding}"
        
        # FALLBACK (should rarely be used now)
        else:
            # Analyze the generated_text to infer the task
            if len(generated_text) < len(grounding) * 0.3:
                # Likely summarization
                return f"Please provide a concise summary of the following text:\n\n{truncated_grounding}"
            elif "?" in generated_text:
                # Likely question answering
                return f"Based on the following information, please answer relevant questions or provide key insights:\n\n{truncated_grounding}"
            else:
                # Text continuation/expansion
                return f"Please provide additional relevant information or expand on the following context:\n\n{truncated_grounding}"

def main():
    parser = argparse.ArgumentParser(description='Generate responses for TRUE benchmark evaluation')
    parser.add_argument('--data_dir', type=str,
                       default='/data/hector/datasets/true_datasets',
                       help='Directory containing TRUE datasets')
    parser.add_argument('--output_dir', type=str,
                       default='/data/hector/generated_responses',
                       help='Directory to save generated responses')
    parser.add_argument('--server_url', type=str,
                       default='http://localhost:8888',
                       help='URL of your model server')
    parser.add_argument('--datasets', nargs='+',
                       help='Specific datasets to process (leave empty for all)')
    parser.add_argument('--with_ads', action='store_true',
                       help='Generate responses with ads')
    parser.add_argument('--without_ads', action='store_true',
                       help='Generate responses without ads')
    parser.add_argument('--both', action='store_true',
                       help='Generate both with and without ads (default)')
    parser.add_argument('--force', action='store_true',
                       help='Force regeneration even if output files exist')
    
    args = parser.parse_args()
    
    # Default to both if nothing specified
    if not args.with_ads and not args.without_ads and not args.both:
        args.both = True
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = TRUEResponseGenerator(args.server_url)
    
    # Override completion check if force flag is used
    if args.force:
        print("🔄 Force mode enabled - will regenerate all files")
        generator.is_output_complete = lambda *args: False
    
    # Define datasets to process (matching your actual files)
    all_datasets = [
        ('begin_dev_download.csv', 'BEGIN'),
        ('dialfact_valid_download.csv', 'DialFact'),
        ('frank_valid_download.csv', 'FRANK'),
        ('mnbm_download.csv', 'MNBM'),
        ('q2_download.csv', 'Q2'),
        ('qags_cnndm_download.csv', 'QAGS_CNNDM'),
        ('qags_xsum_download.csv', 'QAGS_XSum'),
        ('paws_download.csv', 'PAWS'),
        ('summeval_download.csv', 'SummEval'),
        ('fever_dev_download.csv', 'FEVER'),
        ('vitc_dev_download.csv', 'VitaminC')  # Added VitaminC dataset
    ]
    
    # Filter datasets if specified
    if args.datasets:
        all_datasets = [(csv, name) for csv, name in all_datasets 
                       if name.lower() in [d.lower() for d in args.datasets]]
    
    print(f"🚀 Starting response generation for {len(all_datasets)} datasets")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Note: Datasets with >1000 rows will be limited to first 1000")
    print(f"💾 Files will be written incrementally (line by line)")
    if not args.force:
        print(f"⏭️  Complete files will be skipped automatically")
    
    generated_files = []
    skipped_files = []
    
    # Process each dataset
    for csv_file, dataset_name in all_datasets:
        input_path = os.path.join(args.data_dir, csv_file)
        
        if not os.path.exists(input_path):
            print(f"⚠️  Skipping {dataset_name}: file not found at {input_path}")
            continue
        
        # Generate without ads
        if args.without_ads or args.both:
            try:
                output_file = generator.process_true_dataset(
                    input_path, args.output_dir, dataset_name, with_ads=False
                )
                if output_file:
                    generated_files.append(output_file)
            except Exception as e:
                print(f"❌ Error processing {dataset_name} without ads: {e}")
        
        # Generate with ads
        if args.with_ads or args.both:
            try:
                output_file = generator.process_true_dataset(
                    input_path, args.output_dir, dataset_name, with_ads=True
                )
                if output_file:
                    generated_files.append(output_file)
            except Exception as e:
                print(f"❌ Error processing {dataset_name} with ads: {e}")
    
    print(f"\n🎉 Response generation completed!")
    print(f"📊 Generated/updated {len(generated_files)} files:")
    for file in generated_files:
        print(f"   - {file}")

if __name__ == "__main__":
    main()
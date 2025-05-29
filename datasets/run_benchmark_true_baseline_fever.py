#!/usr/bin/env python3
"""
Debug FEVER dataset evaluation with TRUE benchmark models
Focus on identifying issues with the TRUE model evaluation process
"""

import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, BitsAndBytesConfig
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import time
import traceback

class FEVERTRUEDebugEvaluator:
    def __init__(self, model_name="google/t5_xxl_true_nli_mixture", use_quantization=True, quantization_bits=8):
        """
        Initialize TRUE model evaluator with extensive debugging for FEVER.
        """
        self.model_name = model_name
        print(f"[DEBUG] Loading TRUE evaluator model: {model_name}")
        print(f"[DEBUG] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[DEBUG] GPU: {torch.cuda.get_device_name()}")
            print(f"[DEBUG] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load tokenizer
        try:
            print(f"[DEBUG] Loading tokenizer...")
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            print(f"[DEBUG] Tokenizer loaded successfully")
            print(f"[DEBUG] Tokenizer vocab size: {len(self.tokenizer)}")
        except Exception as e:
            print(f"[ERROR] Failed to load tokenizer: {e}")
            raise
        
        # Configure model loading with debugging
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
        
        if use_quantization:
            print(f"[DEBUG] Configuring {quantization_bits}-bit quantization...")
            if quantization_bits == 8:
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=False,
                )
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            model_kwargs["quantization_config"] = bnb_config
        
        # Load model with error handling
        try:
            print(f"[DEBUG] Loading model with config: {model_kwargs}")
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name, **model_kwargs
            )
            print(f"[DEBUG] Model loaded successfully")
            print(f"[DEBUG] Model device: {self.model.device}")
            print(f"[DEBUG] Model dtype: {self.model.dtype}")
            
            # Test the model
            self._comprehensive_model_test()
            
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            traceback.print_exc()
            raise
    
    def _comprehensive_model_test(self):
        """Comprehensive model testing for FEVER-specific scenarios."""
        print(f"\n[DEBUG] Running comprehensive model tests...")
        
        # Test 1: Basic functionality
        try:
            print(f"[DEBUG] Test 1: Basic TRUE model functionality")
            test_grounding = "The sky is blue during the day."
            test_hypothesis = "The sky appears blue in daylight."
            
            score = self.evaluate_consistency(test_grounding, test_hypothesis, debug=True)
            print(f"[DEBUG] Test 1 PASSED. Consistency score: {score}")
            
        except Exception as e:
            print(f"[ERROR] Test 1 FAILED: {e}")
            traceback.print_exc()
        
        # Test 2: FEVER-style example
        try:
            print(f"\n[DEBUG] Test 2: FEVER-style fact verification")
            fever_grounding = "Barack Obama was the 44th President of the United States, serving from 2009 to 2017."
            fever_hypothesis = "Barack Obama served as President."
            
            score = self.evaluate_consistency(fever_grounding, fever_hypothesis, debug=True)
            print(f"[DEBUG] Test 2 PASSED. FEVER-style score: {score}")
            
        except Exception as e:
            print(f"[ERROR] Test 2 FAILED: {e}")
            traceback.print_exc()
        
        # Test 3: Edge cases
        try:
            print(f"\n[DEBUG] Test 3: Edge cases")
            
            # Empty text
            score1 = self.evaluate_consistency("", "test", debug=True)
            print(f"[DEBUG] Empty grounding score: {score1}")
            
            # Very long text
            long_text = "This is a test sentence. " * 100
            score2 = self.evaluate_consistency(long_text, "This is about testing.", debug=True)
            print(f"[DEBUG] Long text score: {score2}")
            
            print(f"[DEBUG] Test 3 PASSED")
            
        except Exception as e:
            print(f"[ERROR] Test 3 FAILED: {e}")
            traceback.print_exc()
    
    def evaluate_consistency(self, grounding_text: str, generated_text: str, debug: bool = False) -> float:
        """
        Evaluate factual consistency with extensive debugging.
        """
        if debug:
            print(f"[DEBUG] Evaluating consistency:")
            print(f"[DEBUG]   Grounding length: {len(grounding_text)}")
            print(f"[DEBUG]   Generated length: {len(generated_text)}")
            print(f"[DEBUG]   Grounding preview: {grounding_text}...")
            print(f"[DEBUG]   Generated preview: {generated_text}...")
        
        # Handle empty inputs
        if not grounding_text.strip() or not generated_text.strip():
            if debug:
                print(f"[DEBUG] Empty input detected, returning neutral score")
            return 0.5
        
        # Format input according to TRUE benchmark format
        input_text = f"premise: {grounding_text} hypothesis: {generated_text}"
        
        if debug:
            print(f"[DEBUG] Formatted input length: {len(input_text)}")
            print(f"[DEBUG] Formatted input preview: {input_text}...")
        
        try:
            # Tokenize input with debugging
            if debug:
                print(f"[DEBUG] Tokenizing input...")
            
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=2048,  # TRUE models support up to 2048 tokens
                truncation=True,
                padding=True
            )
            
            if debug:
                print(f"[DEBUG] Input shape: {inputs['input_ids'].shape}")
                print(f"[DEBUG] Input tokens: {inputs['input_ids'].shape[1]}")
                print(f"[DEBUG] Moving inputs to device: {self.model.device}")
            
            inputs = inputs.to(self.model.device)
            
            # Generate prediction with debugging
            if debug:
                print(f"[DEBUG] Generating prediction...")
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=2,  # TRUE models output single token
                    num_beams=1,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            if debug:
                print(f"[DEBUG] Output shape: {outputs.shape}")
                print(f"[DEBUG] Raw output tokens: {outputs[0].tolist()}")
            
            # Decode result
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if debug:
                print(f"[DEBUG] Decoded result: '{result}'")
                print(f"[DEBUG] Result stripped: '{result.strip()}'")
            
            # Parse TRUE model output with debugging
            result_clean = result.strip()
            
            if result_clean == "1":
                score = 1.0  # Factually consistent
            elif result_clean == "0":  
                score = 0.0  # Factually inconsistent
            else:
                # Fallback parsing for unexpected outputs
                if debug:
                    print(f"[DEBUG] Unexpected output, attempting fallback parsing...")
                
                if "1" in result:
                    score = 0.8
                elif "0" in result:
                    score = 0.2
                else:
                    score = 0.5  # Uncertain
                    
                if debug:
                    print(f"[DEBUG] Fallback parsing result: {score}")
            
            if debug:
                print(f"[DEBUG] Final score: {score}")
            
            return score
                    
        except Exception as e:
            print(f"[ERROR] Evaluation error: {e}")
            if debug:
                traceback.print_exc()
            return 0.5  # Return neutral score on error
    
    def debug_dataset_structure(self, csv_file: str):
        """Debug the structure of the dataset file."""
        print(f"\n[DEBUG] Analyzing dataset structure: {csv_file}")
        
        if not Path(csv_file).exists():
            print(f"[ERROR] File does not exist: {csv_file}")
            return False
        
        print(f"[DEBUG] File exists, size: {Path(csv_file).stat().st_size} bytes")
        
        try:
            # Read first few lines
            with open(csv_file, 'r', encoding='utf-8') as f:
                first_lines = [f.readline().strip() for _ in range(3)]
            
            print(f"[DEBUG] First 3 lines:")
            for i, line in enumerate(first_lines):
                print(f"  Line {i+1}: {line[:150]}...")
            
            # Load with pandas
            df = pd.read_csv(csv_file)
            print(f"[DEBUG] Successfully loaded DataFrame with {len(df)} rows")
            print(f"[DEBUG] Columns: {list(df.columns)}")
            
            # Check column content
            for col in df.columns:
                print(f"[DEBUG] Column '{col}':")
                print(f"  - Type: {df[col].dtype}")
                print(f"  - Non-null: {df[col].notna().sum()}/{len(df)}")
                if df[col].dtype == 'object':
                    sample_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else "N/A"
                    print(f"  - Sample: {str(sample_val)[:100]}...")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to analyze dataset: {e}")
            traceback.print_exc()
            return False
    
    def evaluate_fever_dataset_debug(self, csv_file: str, max_samples: int = 5) -> dict:
        """
        Debug FEVER dataset evaluation with detailed logging.
        """
        print(f"\n[DEBUG] Starting FEVER dataset debug evaluation")
        print(f"[DEBUG] File: {csv_file}")
        print(f"[DEBUG] Max samples: {max_samples}")
        
        # Debug dataset structure first
        if not self.debug_dataset_structure(csv_file):
            return None
        
        # Load dataset
        try:
            df = pd.read_csv(csv_file)
            original_length = len(df)
            
            # Limit for debugging
            df = df.head(max_samples)
            print(f"[DEBUG] Processing {len(df)} samples (original: {original_length})")
            
        except Exception as e:
            print(f"[ERROR] Failed to load dataset: {e}")
            return None
        
        # Identify columns
        print(f"\n[DEBUG] Identifying required columns...")
        
        # Look for grounding column
        grounding_candidates = ['grounding', 'premise', 'evidence', 'context']
        grounding_col = None
        for col in grounding_candidates:
            if col in df.columns:
                grounding_col = col
                break
        
        # Look for generated text column  
        generated_candidates = ['generated_text', 'generated_text_with_ads', 'hypothesis', 'text']
        generated_col = None
        for col in generated_candidates:
            if col in df.columns:
                generated_col = col
                break
        
        # Look for label column
        label_candidates = ['label', 'ground_truth', 'original_label']
        label_col = None
        for col in label_candidates:
            if col in df.columns:
                label_col = col
                break
        
        print(f"[DEBUG] Column mapping:")
        print(f"  Grounding: {grounding_col}")
        print(f"  Generated: {generated_col}")
        print(f"  Label: {label_col}")
        
        if not grounding_col or not generated_col:
            print(f"[ERROR] Missing required columns")
            print(f"[ERROR] Available columns: {list(df.columns)}")
            return None
        
        # Process samples with detailed debugging
        results = []
        
        for idx, row in df.iterrows():
            print(f"\n[DEBUG] Processing sample {idx + 1}/{len(df)}")
            
            try:
                grounding = str(row[grounding_col])
                generated = str(row[generated_col])
                label = row[label_col] if label_col else None
                
                print(f"[DEBUG] Sample {idx}:")
                print(f"  Grounding length: {len(grounding)}")
                print(f"  Generated length: {len(generated)}")
                print(f"  Label: {label}")
                print(f"  Grounding: {grounding}")
                print(f"  Generated: {generated}")
                
                # Skip if texts are too short
                if len(grounding.strip()) < 10 or len(generated.strip()) < 10:
                    print(f"[DEBUG] Sample {idx}: Skipping - text too short")
                    score = 0.5
                    error_msg = "Text too short"
                else:
                    # Evaluate with debugging
                    print(f"[DEBUG] Sample {idx}: Evaluating consistency...")
                    score = self.evaluate_consistency(grounding, generated, debug=True)
                    error_msg = ""
                
                results.append({
                    'index': idx,
                    'grounding_length': len(grounding),
                    'generated_length': len(generated),
                    'label': label,
                    'consistency_score': score,
                    'error_message': error_msg
                })
                
                print(f"[DEBUG] Sample {idx}: Score = {score}")
                
                # Small delay
                time.sleep(0.5)
                
            except Exception as e:
                print(f"[ERROR] Sample {idx}: Exception occurred: {e}")
                traceback.print_exc()
                
                results.append({
                    'index': idx,
                    'grounding_length': 0,
                    'generated_length': 0,
                    'label': None,
                    'consistency_score': 0.5,
                    'error_message': f"Exception: {str(e)}"
                })
        
        # Calculate summary statistics
        scores = [r['consistency_score'] for r in results]
        avg_score = np.mean(scores) if scores else 0.0
        
        summary = {
            'dataset': 'FEVER',
            'total_samples': len(results),
            'avg_consistency_score': avg_score,
            'scores': scores,
            'detailed_results': results
        }
        
        print(f"\n[DEBUG] FEVER evaluation completed!")
        print(f"[DEBUG] Average consistency score: {avg_score:.4f}")
        print(f"[DEBUG] Score distribution: {np.histogram(scores, bins=5)[0]}")
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Debug FEVER dataset TRUE evaluation')
    parser.add_argument('--data_file', type=str,
                       help='Path to FEVER CSV file')
    parser.add_argument('--output_dir', type=str, default='fever_true_debug',
                       help='Directory to save debug results')
    parser.add_argument('--model_path', type=str, 
                       default='/data/hector/models/t5_xxl_true_nli',
                       help='Path to local TRUE model directory')
    parser.add_argument('--model_name', type=str, 
                       default='google/t5_xxl_true_nli_mixture',
                       help='HuggingFace model name')
    parser.add_argument('--quantization_bits', type=int, default=8,
                       choices=[4, 8], help='Quantization bits')
    parser.add_argument('--no_quantization', action='store_true',
                       help='Disable quantization')
    parser.add_argument('--max_samples', type=int, default=5,
                       help='Maximum samples for debugging')
    
    args = parser.parse_args()
    
    print(f"[DEBUG] FEVER TRUE Evaluation Debug Tool")
    print(f"[DEBUG] Data file: {args.data_file}")
    print(f"[DEBUG] Output directory: {args.output_dir}")
    print(f"[DEBUG] Max samples: {args.max_samples}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Determine model path
    model_path = Path(args.model_path)
    if model_path.exists():
        print(f"[DEBUG] Using local model: {model_path}")
        model_to_load = str(model_path)
    else:
        print(f"[DEBUG] Using HuggingFace model: {args.model_name}")
        model_to_load = args.model_name
    
    # Initialize evaluator
    try:
        evaluator = FEVERTRUEDebugEvaluator(
            model_name=model_to_load,
            use_quantization=not args.no_quantization,
            quantization_bits=args.quantization_bits
        )
    except Exception as e:
        print(f"[ERROR] Failed to initialize evaluator: {e}")
        return
    
    # Find or use specified data file
    if args.data_file:
        data_file = args.data_file
    else:
        # Look for FEVER files in common locations
        possible_files = [
            '/data/hector/baseline_with_ads/fever_baseline_with_ads.csv'
        ]
        
        data_file = None
        for file_path in possible_files:
            if Path(file_path).exists():
                data_file = file_path
                break
        
        if not data_file:
            print(f"[ERROR] No FEVER data file found. Tried:")
            for fp in possible_files:
                print(f"  - {fp}")
            print(f"Please specify with --data_file")
            return
    
    print(f"[DEBUG] Using data file: {data_file}")
    
    # Run debug evaluation
    try:
        result = evaluator.evaluate_fever_dataset_debug(data_file, args.max_samples)
        
        if result:
            # Save results
            output_file = output_dir / "fever_true_debug_results.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            
            print(f"\n[SUCCESS] Debug results saved to: {output_file}")
            print(f"[SUCCESS] Average score: {result['avg_consistency_score']:.4f}")
            
        else:
            print(f"[ERROR] Debug evaluation failed")
    
    except Exception as e:
        print(f"[ERROR] Debug evaluation exception: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
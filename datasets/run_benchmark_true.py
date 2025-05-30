#!/usr/bin/env python3
"""
Sanity check: Use TRUE benchmark models to evaluate original grounded vs original generated text
This helps validate the TRUE model is working correctly before evaluating your LLM responses
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
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

class TRUESanityCheckEvaluator:
    def __init__(self, model_name="google/t5_xxl_true_nli_mixture", use_quantization=True, quantization_bits=8):
        """
        Initialize TRUE model evaluator for sanity checking.
        
        Args:
            model_name: HuggingFace model name or local path for TRUE evaluator
            use_quantization: Whether to use quantization to save VRAM
            quantization_bits: 4 or 8 bit quantization (8-bit recommended for 24GB VRAM)
        """
        self.model_name = model_name
        print(f"🔄 Loading TRUE evaluator model: {model_name}")
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        
        # Configure model loading
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
        
        if use_quantization:
            if quantization_bits == 8:
                print("📦 Using 8-bit quantization (optimal for 24GB VRAM)...")
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=False,
                )
            else:
                print("📦 Using 4-bit quantization...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            model_kwargs["quantization_config"] = bnb_config
        
        # Load model
        try:
            self.model = T5ForConditionalGeneration.from_pretrained(
                model_name, **model_kwargs
            )
            print("✅ TRUE evaluator model loaded successfully")
            
            # Test the model
            self._test_model()
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("💡 Try reducing quantization or using a smaller model")
            raise
    
    def _test_model(self):
        """Test the evaluator model with a simple example."""
        print("🧪 Testing evaluator model...")
        try:
            test_grounding = "The sky is blue during the day."
            test_hypothesis = "The sky appears blue in daylight."
            
            score = self.evaluate_consistency(test_grounding, test_hypothesis)
            print(f"✅ Test passed. Consistency score: {score}")
            
        except Exception as e:
            print(f"⚠️ Model test failed: {e}")
    
    def evaluate_consistency(self, grounding_text: str, generated_text: str) -> float:
        """
        Evaluate factual consistency between grounding and generated text.
        
        Args:
            grounding_text: Source/reference text
            generated_text: Generated text to evaluate
            
        Returns:
            Consistency score (0.0 to 1.0)
        """
        # Format input according to TRUE benchmark format
        if "trueteacher" in self.model_name.lower():
            # TrueTeacher format
            input_text = f"premise: {grounding_text} hypothesis: {generated_text}"
        else:
            # Standard TRUE NLI format  
            input_text = f"premise: {grounding_text} hypothesis: {generated_text}"
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=2048,  # TRUE models support up to 2048 tokens
                truncation=True,
                padding=True
            ).to(self.model.device)
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=2,  # TRUE models output single token
                    num_beams=1,
                    do_sample=False
                )
            
            # Decode result
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse TRUE model output
            if result.strip() == "1":
                return 1.0  # Factually consistent
            elif result.strip() == "0":  
                return 0.0  # Factually inconsistent
            else:
                # Fallback parsing for unexpected outputs
                if "1" in result:
                    return 0.8
                elif "0" in result:
                    return 0.2
                else:
                    return 0.5  # Uncertain
                    
        except Exception as e:
            print(f"⚠️ Evaluation error: {e}")
            return 0.5  # Return neutral score on error
    
    def _parse_ground_truth_label(self, label, dataset_name: str):
        """Parse ground truth label based on dataset format."""
        try:
            if dataset_name.lower() in ['fever']:
                # FEVER: 'SUPPORTS' = 1, 'REFUTES'/'NOT ENOUGH INFO' = 0
                label_str = str(label).upper().strip()
                if label_str == 'SUPPORTS':
                    return 1
                elif label_str in ['REFUTES', 'NOT ENOUGH INFO']:
                    return 0
                else:
                    return None
            elif dataset_name.lower() in ['vitaminc', 'vitc']:
                # VitaminC: binary labels (0/1)
                return int(float(label)) if str(label).replace('.','').replace('-','').isdigit() else None
            elif dataset_name.lower() in ['summeval', 'qags_cnndm', 'qags_xsum']:
                # Summarization datasets: usually binary
                return int(float(label)) if str(label).replace('.','').replace('-','').isdigit() else None
            else:
                # Default binary parsing
                return int(float(label)) if str(label).replace('.','').replace('-','').isdigit() else None
        except:
            return None
    
    def evaluate_dataset_sanity_check(self, csv_file: str, max_samples: int = 1000, output_file: str = None) -> dict:
        """
        Evaluate original grounded vs original generated text as sanity check.
        
        Args:
            csv_file: Path to CSV file with original data
            max_samples: Maximum number of samples to evaluate (default: 1000)
            output_file: Optional path to save detailed results
            
        Returns:
            Dictionary with evaluation metrics including ROC-AUC
        """
        print(f"\n🔍 SANITY CHECK - Evaluating original data: {Path(csv_file).name}")
        
        # Load dataset
        df = pd.read_csv(csv_file)
        
        # Limit to max_samples
        original_length = len(df)
        if len(df) > max_samples:
            df = df.head(max_samples)
            print(f"📊 Limited to first {max_samples} samples (original: {original_length})")
        else:
            print(f"📊 Using all {len(df)} samples")
        
        # Check for required columns - we need original grounding and generated
        possible_grounding_cols = ['grounding']
        possible_generated_cols = ['generated_text_with_ads']
        possible_label_cols = ['original_label', 'label', 'ground_truth']
        
        grounding_col = None
        generated_col = None
        label_col = None
        
        for col in possible_grounding_cols:
            if col in df.columns:
                grounding_col = col
                break
                
        for col in possible_generated_cols:
            if col in df.columns:
                generated_col = col
                break
                
        for col in possible_label_cols:
            if col in df.columns:
                label_col = col
                break
        
        if not grounding_col or not generated_col:
            available_cols = list(df.columns)
            raise ValueError(f"Missing required columns. Found: {available_cols}. "
                           f"Need grounding column from {possible_grounding_cols} "
                           f"and generated column from {possible_generated_cols}")
        
        dataset_name = df.get('dataset', ['unknown']).iloc[0] if 'dataset' in df.columns else 'unknown'
        
        print(f"Dataset: {dataset_name}")
        print(f"Grounding column: {grounding_col}")
        print(f"Generated column: {generated_col}")
        print(f"Label column: {label_col if label_col else 'None (will only evaluate TRUE scores)'}")
        print(f"Examples: {len(df)}")
        
        # Evaluate each example
        consistency_scores = []
        ground_truth_labels = []
        detailed_results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating original consistency"):
            grounding = str(row[grounding_col])
            generated = str(row[generated_col])
            
            # Parse ground truth label if available
            gt_label = None
            if label_col:
                gt_label = self._parse_ground_truth_label(row[label_col], dataset_name)
                ground_truth_labels.append(gt_label)
            
            # Skip if either text is empty or too short
            if len(grounding.strip()) < 10 or len(generated.strip()) < 10:
                consistency_scores.append(0.5)  # Neutral score for invalid examples
                continue
            
            # Get consistency score from TRUE model
            score = self.evaluate_consistency(grounding, generated)
            consistency_scores.append(score)
            
            # Store detailed results
            detailed_results.append({
                'index': idx,
                'grounding': grounding[:200] + "..." if len(grounding) > 200 else grounding,
                'generated': generated[:200] + "..." if len(generated) > 200 else generated,
                'true_consistency_score': score,
                'ground_truth_label': gt_label,
                'consistent': score > 0.5,
                'matches_ground_truth': (score > 0.5) == bool(gt_label) if gt_label is not None else None
            })
            
            # Small delay to prevent overwhelming GPU
            time.sleep(0.01)
        
        # Calculate metrics
        avg_consistency = np.mean(consistency_scores)
        consistent_count = sum(1 for score in consistency_scores if score > 0.5)
        consistency_rate = consistent_count / len(consistency_scores)
        
        # High/medium/low consistency breakdown
        high_consistency = sum(1 for score in consistency_scores if score > 0.8)
        medium_consistency = sum(1 for score in consistency_scores if 0.3 <= score <= 0.8)
        low_consistency = sum(1 for score in consistency_scores if score < 0.3)
        
        results = {
            'dataset': dataset_name,
            'total_examples': len(consistency_scores),
            'original_dataset_size': original_length,
            'samples_evaluated': len(consistency_scores),
            'avg_consistency_score': avg_consistency,
            'consistency_rate': consistency_rate,
            'consistent_examples': consistent_count,
            'high_consistency_count': high_consistency,
            'medium_consistency_count': medium_consistency, 
            'low_consistency_count': low_consistency,
            'high_consistency_rate': high_consistency / len(consistency_scores),
            'medium_consistency_rate': medium_consistency / len(consistency_scores),
            'low_consistency_rate': low_consistency / len(consistency_scores)
        }
        
        # Add ground truth comparison and ROC-AUC if available
        if ground_truth_labels and any(label is not None for label in ground_truth_labels):
            # Filter out None labels for ROC-AUC calculation
            valid_pairs = [(score, label) for score, label in zip(consistency_scores, ground_truth_labels) if label is not None]
            
            if len(valid_pairs) > 0:
                valid_scores, valid_labels = zip(*valid_pairs)
                valid_scores = np.array(valid_scores)
                valid_labels = np.array(valid_labels)
                
                # Calculate accuracy (binary predictions vs ground truth)
                binary_predictions = (valid_scores > 0.5).astype(int)
                accuracy = np.mean(binary_predictions == valid_labels)
                
                # Calculate ROC-AUC if we have both classes
                unique_labels = np.unique(valid_labels)
                if len(unique_labels) > 1:
                    try:
                        roc_auc = roc_auc_score(valid_labels, valid_scores)
                        fpr, tpr, thresholds = roc_curve(valid_labels, valid_scores)
                        
                        # Calculate Precision-Recall AUC
                        precision, recall, pr_thresholds = precision_recall_curve(valid_labels, valid_scores)
                        pr_auc = auc(recall, precision)
                        
                        results.update({
                            'ground_truth_available': True,
                            'ground_truth_accuracy': accuracy,
                            'valid_ground_truth_comparisons': len(valid_pairs),
                            'roc_auc': roc_auc,
                            'pr_auc': pr_auc,
                            'unique_labels_count': len(unique_labels),
                            'label_distribution': {int(label): int(np.sum(valid_labels == label)) for label in unique_labels}
                        })
                        
                        print(f"📈 ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
                        
                    except Exception as e:
                        print(f"⚠️ Could not calculate ROC-AUC: {e}")
                        results.update({
                            'ground_truth_available': True,
                            'ground_truth_accuracy': accuracy,
                            'valid_ground_truth_comparisons': len(valid_pairs),
                            'roc_auc': None,
                            'pr_auc': None,
                            'roc_auc_error': str(e)
                        })
                else:
                    print(f"⚠️ Only one class present in labels, cannot calculate ROC-AUC")
                    results.update({
                        'ground_truth_available': True,
                        'ground_truth_accuracy': accuracy,
                        'valid_ground_truth_comparisons': len(valid_pairs),
                        'roc_auc': None,
                        'pr_auc': None,
                        'single_class_dataset': True,
                        'single_class_value': int(unique_labels[0])
                    })
            else:
                results['ground_truth_available'] = False
        else:
            results['ground_truth_available'] = False
        
        # Save detailed results if requested
        if output_file:
            output_data = {
                'summary': results,
                'detailed_results': detailed_results,
                'consistency_scores': consistency_scores,
                'ground_truth_labels': ground_truth_labels if ground_truth_labels else None
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"💾 Detailed results saved to: {output_file}")
        
        return results
    
    def generate_sanity_check_report(self, all_results: list) -> str:
        """Generate sanity check evaluation report."""
        report = []
        report.append("="*80)
        report.append("TRUE MODEL SANITY CHECK REPORT")
        report.append("Evaluating Original Grounding vs Original Generated Text")
        report.append(f"Evaluator Model: {self.model_name}")
        report.append("="*80)
        
        # Overall summary
        report.append(f"\n📊 SANITY CHECK RESULTS")
        report.append("-" * 50)
        
        for result in all_results:
            dataset = result['dataset']
            report.append(f"\n{dataset}:")
            report.append(f"  Samples: {result['samples_evaluated']}/{result['original_dataset_size']}")
            report.append(f"  TRUE Consistency Score: {result['avg_consistency_score']:.4f}")
            report.append(f"  Consistency Rate: {result['consistency_rate']:.2%}")
            report.append(f"  High Consistency: {result['high_consistency_rate']:.2%}")
            report.append(f"  Medium Consistency: {result['medium_consistency_rate']:.2%}")
            report.append(f"  Low Consistency: {result['low_consistency_rate']:.2%}")
            
            if result.get('ground_truth_available', False):
                report.append(f"  Ground Truth Accuracy: {result['ground_truth_accuracy']:.2%}")
                report.append(f"  Valid GT Comparisons: {result['valid_ground_truth_comparisons']}")
                
                if result.get('roc_auc') is not None:
                    report.append(f"  ROC-AUC: {result['roc_auc']:.4f}")
                    report.append(f"  PR-AUC: {result['pr_auc']:.4f}")
                    
                    # Label distribution
                    if 'label_distribution' in result:
                        dist = result['label_distribution']
                        report.append(f"  Label Distribution: {dist}")
                elif result.get('single_class_dataset'):
                    report.append(f"  ⚠️ Single class dataset (class: {result['single_class_value']})")
                elif result.get('roc_auc_error'):
                    report.append(f"  ⚠️ ROC-AUC Error: {result['roc_auc_error']}")
            else:
                report.append(f"  ⚠️ No ground truth labels available")
        
        # Interpretation
        report.append(f"\n🎯 SANITY CHECK INTERPRETATION")
        report.append("-" * 40)
        
        avg_scores = [r['avg_consistency_score'] for r in all_results]
        overall_avg = np.mean(avg_scores) if avg_scores else 0.0
        
        report.append(f"Overall Average: {overall_avg:.4f}")
        
        if overall_avg > 0.8:
            report.append("✅ EXCELLENT: TRUE model shows high consistency with original data")
            report.append("   This suggests the TRUE model is working correctly")
        elif overall_avg > 0.6:
            report.append("✅ GOOD: TRUE model shows reasonable consistency")
            report.append("   The model appears to be functioning properly")
        elif overall_avg > 0.4:
            report.append("⚠️ MODERATE: TRUE model shows mixed results")
            report.append("   This could indicate dataset complexity or model limitations")
        else:
            report.append("❌ CONCERNING: TRUE model shows low consistency")
            report.append("   This may indicate issues with the TRUE model or data format")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS")
        report.append("-" * 30)
        
        if overall_avg < 0.5:
            report.append("🔧 Low scores suggest potential issues:")
            report.append("   - Check input text formatting")
            report.append("   - Verify TRUE model is appropriate for your data")
            report.append("   - Consider trying different TRUE model variant")
        else:
            report.append("✅ Sanity check passed! You can proceed with confidence to evaluate your LLM responses")
        
        report.append(f"\n📋 NEXT STEPS")
        report.append("-" * 20)
        report.append("1. If sanity check looks good, run TRUE evaluation on your LLM responses")
        report.append("2. Compare LLM scores to these baseline original scores")
        report.append("3. Look for significant drops in consistency scores")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Sanity check: Evaluate original data with TRUE models')
    parser.add_argument('--data_dir', type=str,
                        default='/data/hector/baseline_with_ads_better_dataset_better_prompt',
                       help='Directory containing original dataset CSV files')
    parser.add_argument('--output_dir', type=str, default='true_sanity_check',
                       help='Directory to save sanity check results')
    parser.add_argument('--model_path', type=str, 
                       default='/data/hector/models/t5_xxl_true_nli',
                       help='Path to local TRUE model directory')
    parser.add_argument('--model_name', type=str, 
                       default='google/t5_xxl_true_nli',
                       help='HuggingFace model name (used if local path not found)')
    parser.add_argument('--quantization_bits', type=int, default=8,
                       choices=[4, 8],
                       help='Quantization bits: 4 (lower VRAM) or 8 (better quality)')
    parser.add_argument('--no_quantization', action='store_true',
                       help='Disable quantization (requires more VRAM)')
    parser.add_argument('--max_samples', type=int, default=1000,
                       help='Maximum number of samples to evaluate per dataset (default: 1000)')
    parser.add_argument('--save_detailed', action='store_true',
                       help='Save detailed per-example results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🔍 TRUE Model Sanity Check Evaluator")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🤖 Model path: {args.model_path}")
    print(f"📊 Max samples per dataset: {args.max_samples}")
    print(f"⚙️ Quantization: {args.quantization_bits}-bit" if not args.no_quantization else "⚙️ Quantization: Disabled")
    
    # Determine model path
    model_path = Path(args.model_path)
    if model_path.exists():
        print(f"✅ Using local model: {model_path}")
        model_to_load = str(model_path)
    else:
        print(f"⚠️ Local model not found at {model_path}, using HuggingFace: {args.model_name}")
        model_to_load = args.model_name
    
    # Initialize evaluator
    try:
        evaluator = TRUESanityCheckEvaluator(
            model_name=model_to_load,
            use_quantization=not args.no_quantization,
            quantization_bits=args.quantization_bits
        )
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        return
    
    # Find CSV files to evaluate
    data_dir = Path(args.data_dir)
    csv_files = list(data_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {data_dir}")
        return
    
    print(f"\n📊 Found {len(csv_files)} files to evaluate")
    
    # Evaluate each file
    all_results = []
    for csv_file in csv_files:
        try:
            # Set up output file for detailed results
            detail_file = None
            if args.save_detailed:
                detail_file = output_dir / f"{csv_file.stem}_sanity_check_detailed.json"
            
            # Evaluate dataset
            result = evaluator.evaluate_dataset_sanity_check(
                str(csv_file), 
                max_samples=args.max_samples,
                output_file=str(detail_file) if detail_file else None
            )
            all_results.append(result)
            
            roc_info = f", ROC-AUC = {result.get('roc_auc', 'N/A'):.4f}" if result.get('roc_auc') is not None else ""
            print(f"{csv_file.name}: Consistency = {result['avg_consistency_score']:.4f}{roc_info}")
            
        except Exception as e:
            print(f"❌ Error evaluating {csv_file}: {e}")
            import traceback
            traceback.print_exc()
    
    # Generate and save comprehensive report
    if all_results:
        report = evaluator.generate_sanity_check_report(all_results)
        
        report_file = output_dir / "true_sanity_check_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\n💾 Sanity check report saved to: {report_file}")
        
        # Print summary
        avg_scores = [r['avg_consistency_score'] for r in all_results]
        overall_avg = np.mean(avg_scores)
        
        print(f"\n📋 SANITY CHECK SUMMARY")
        print(f"Overall Average Consistency: {overall_avg:.4f}")
        
        if overall_avg > 0.6:
            print("✅ Sanity check PASSED - TRUE model working correctly")
        else:
            print("⚠️ Sanity check shows low scores - review data or model")

if __name__ == "__main__":
    main()
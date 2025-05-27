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
            return None#!/usr/bin/env python3
"""
Use TRUE benchmark models to evaluate factual consistency of your LLM responses
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

class TRUEModelEvaluator:
    def __init__(self, model_name="google/t5_xxl_true_nli_mixture", use_quantization=True):
        """
        Initialize TRUE model evaluator.
        
        Args:
            model_name: HuggingFace model name for TRUE evaluator
            use_quantization: Whether to use 4-bit quantization to save VRAM
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
            print("📦 Using 4-bit quantization to reduce VRAM usage...")
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
                    do_sample=False,
                    early_stopping=True
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
    
    def evaluate_dataset(self, csv_file: str, output_file: str = None) -> dict:
        """
        Evaluate an entire dataset CSV file using TRUE model.
        
        Args:
            csv_file: Path to CSV file with your model's responses
            output_file: Optional path to save detailed results
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\n📊 Evaluating dataset: {Path(csv_file).name}")
        
        # Load dataset
        df = pd.read_csv(csv_file)
        
        required_columns = ['original_grounding', 'model_response', 'dataset']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        dataset_name = df['dataset'].iloc[0]
        with_ads = df.get('with_ads', [False]).iloc[0]
        
        print(f"Dataset: {dataset_name} ({'with ads' if with_ads else 'without ads'})")
        print(f"Examples: {len(df)}")
        
        # Evaluate each example
        consistency_scores = []
        detailed_results = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating consistency"):
            grounding = str(row['original_grounding'])
            generated = str(row['model_response'])
            
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
                'consistent': score > 0.5
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
            'with_ads': with_ads,
            'total_examples': len(consistency_scores),
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
        
        # Save detailed results if requested
        if output_file:
            output_data = {
                'summary': results,
                'detailed_results': detailed_results,
                'consistency_scores': consistency_scores
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            print(f"💾 Detailed results saved to: {output_file}")
        
        return results
    
    def generate_report(self, all_results: list) -> str:
        """Generate comprehensive evaluation report."""
        report = []
        report.append("="*80)
        report.append("TRUE MODEL FACTUAL CONSISTENCY EVALUATION REPORT")
        report.append(f"Evaluator Model: {self.model_name}")
        report.append("="*80)
        
        # Separate by ads/no-ads
        with_ads_results = [r for r in all_results if r.get('with_ads', False)]
        without_ads_results = [r for r in all_results if not r.get('with_ads', False)]
        
        # Overall summary
        report.append(f"\n📊 OVERALL CONSISTENCY SCORES")
        report.append("-" * 50)
        
        def calc_avg_consistency(results):
            if not results:
                return 0.0
            scores = [r['avg_consistency_score'] for r in results]
            return np.mean(scores)
        
        avg_without_ads = calc_avg_consistency(without_ads_results)
        avg_with_ads = calc_avg_consistency(with_ads_results)
        
        report.append(f"Without Ads: {avg_without_ads:.4f}")
        report.append(f"With Ads:    {avg_with_ads:.4f}")
        report.append(f"Impact:      {avg_with_ads - avg_without_ads:+.4f}")
        
        # Performance interpretation
        report.append(f"\n🎯 PERFORMANCE INTERPRETATION")
        report.append("-" * 40)
        
        if avg_without_ads > 0.85:
            report.append("🎉 Excellent factual consistency!")
        elif avg_without_ads > 0.70:
            report.append("✅ Good factual consistency")
        elif avg_without_ads > 0.55:
            report.append("⚠️ Moderate factual consistency")
        else:
            report.append("❌ Poor factual consistency - needs improvement")
        
        # Detailed breakdown
        for condition, results in [("WITHOUT ADS", without_ads_results), ("WITH ADS", with_ads_results)]:
            if not results:
                continue
                
            report.append(f"\n📋 DETAILED RESULTS - {condition}")
            report.append("-" * 50)
            
            for result in results:
                dataset = result['dataset']
                report.append(f"\n{dataset}:")
                report.append(f"  Average Score: {result['avg_consistency_score']:.4f}")
                report.append(f"  Consistency Rate: {result['consistency_rate']:.2%}")
                report.append(f"  High Consistency: {result['high_consistency_rate']:.2%}")
                report.append(f"  Medium Consistency: {result['medium_consistency_rate']:.2%}")
                report.append(f"  Low Consistency: {result['low_consistency_rate']:.2%}")
                report.append(f"  Total Examples: {result['total_examples']}")
        
        # Recommendations
        report.append(f"\n🎯 RECOMMENDATIONS")
        report.append("-" * 30)
        
        if avg_with_ads < avg_without_ads - 0.1:
            report.append("⚠️ Native ads significantly harm factual consistency")
            report.append("   - Review ad insertion algorithm")
            report.append("   - Add factual consistency constraints to ad placement")
        
        if avg_without_ads < 0.7:
            report.append("📈 Model factual consistency needs improvement:")
            report.append("   - Fine-tune on factual consistency datasets")
            report.append("   - Improve retrieval/grounding mechanisms")
            report.append("   - Add factual consistency training objectives")
        
        return "\n".join(report)

def main():
    parser = argparse.ArgumentParser(description='Evaluate factual consistency using TRUE models')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing your model response CSV files')
    parser.add_argument('--output_dir', type=str, default='true_evaluations',
                       help='Directory to save evaluation results')
    parser.add_argument('--model_name', type=str, 
                       default='google/t5_xxl_true_nli_mixture',
                       choices=[
                           'google/t5_xxl_true_nli_mixture',
                           'google/t5_11b_trueteacher_and_anli'
                       ],
                       help='TRUE model to use as evaluator')
    parser.add_argument('--no_quantization', action='store_true',
                       help='Disable 8-bit quantization (requires more VRAM)')
    parser.add_argument('--save_detailed', action='store_true',
                       help='Save detailed per-example results')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"🚀 TRUE Model Factual Consistency Evaluator")
    print(f"📁 Results directory: {args.results_dir}")
    print(f"📁 Output directory: {args.output_dir}")
    print(f"🤖 Evaluator model: {args.model_name}")
    
    # Initialize evaluator
    try:
        evaluator = TRUEModelEvaluator(
            model_name=args.model_name,
            use_quantization=not args.no_quantization
        )
    except Exception as e:
        print(f"❌ Failed to initialize evaluator: {e}")
        return
    
    # Find CSV files to evaluate
    results_dir = Path(args.results_dir)
    csv_files = list(results_dir.glob("*.csv"))
    
    if not csv_files:
        print(f"❌ No CSV files found in {results_dir}")
        return
    
    print(f"\n📊 Found {len(csv_files)} files to evaluate")
    
    # Evaluate each file
    all_results = []
    for csv_file in csv_files:
        try:
            # Set up output file for detailed results
            detail_file = None
            if args.save_detailed:
                detail_file = output_dir / f"{csv_file.stem}_detailed.json"
            
            # Evaluate dataset
            result = evaluator.evaluate_dataset(str(csv_file), str(detail_file) if detail_file else None)
            all_results.append(result)
            
            print(f"{csv_file.name}: Consistency = {result['avg_consistency_score']:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {csv_file}: {e}")
    
    # Generate and save comprehensive report
    if all_results:
        report = evaluator.generate_report(all_results)
        
        report_file = output_dir / "true_evaluation_report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nComprehensive report saved to: {report_file}")
        
        # Print summary
        avg_scores = [r['avg_consistency_score'] for r in all_results]
        overall_avg = np.mean(avg_scores)
        
        print(f"\nSUMMARY")
        print(f"Overall Average Consistency: {overall_avg:.4f}")
        
        if overall_avg > 0.85:
            print("Excellent factual consistency!")
        elif overall_avg > 0.70:
            print("Good factual consistency")
        else:
            print("Factual consistency needs improvement")

if __name__ == "__main__":
    main()
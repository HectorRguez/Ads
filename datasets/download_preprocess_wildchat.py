from datasets import load_dataset
from tqdm import tqdm
import json
import random
import re
import string
from collections import Counter

def detect_language(text):
    """
    Simple language detection to filter out non-English text.
    Returns True if text appears to be English, False otherwise.
    """
    if not text or len(text.strip()) < 10:
        return False
    
    # Convert to lowercase for analysis
    text_lower = text.lower()
    
    # Check for non-Latin scripts (Russian, Chinese, Arabic, etc.)
    # Russian Cyrillic range
    if re.search(r'[а-яё]', text_lower):
        return False
    
    # Chinese/Japanese characters
    if re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u309f\u30a0-\u30ff]', text):
        return False
    
    # Arabic script
    if re.search(r'[\u0600-\u06ff]', text):
        return False
    
    # Korean script
    if re.search(r'[\uac00-\ud7af]', text):
        return False
    
    # Greek script
    if re.search(r'[α-ωΑ-Ω]', text):
        return False
    
    # Check for high ratio of non-ASCII characters
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if len(text) > 0 and ascii_chars / len(text) < 0.8:
        return False
    
    # Check for common English words (basic heuristic)
    english_indicators = [
        'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
        'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
        'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those',
        'a', 'an', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
        'her', 'us', 'them', 'my', 'your', 'his', 'our', 'their'
    ]
    
    # Count English indicator words
    words = re.findall(r'\b\w+\b', text_lower)
    if len(words) == 0:
        return False
    
    english_word_count = sum(1 for word in words if word in english_indicators)
    english_ratio = english_word_count / len(words)
    
    # Should have at least 5% common English words
    return english_ratio > 0.05

def is_meaningful_content(text):
    """
    Check if the content is meaningful (not just symbols, URLs, code dumps, etc.)
    """
    if not text or len(text.strip()) < 20:
        return False
    
    # Remove whitespace for analysis
    text_clean = re.sub(r'\s+', ' ', text.strip())
    
    # Check for too many special characters
    special_chars = sum(1 for c in text_clean if not c.isalnum() and c not in ' .,!?;:-()[]{}"\'/\\')
    if len(text_clean) > 0 and special_chars / len(text_clean) > 0.3:
        return False
    
    # Check for excessive repetition
    words = text_clean.lower().split()
    if len(words) > 10:
        word_counts = Counter(words)
        most_common_count = word_counts.most_common(1)[0][1]
        if most_common_count > len(words) * 0.3:  # Same word appears >30% of the time
            return False
    
    # Check for code patterns (basic heuristic)
    code_indicators = ['def ', 'class ', 'import ', 'function ', 'var ', 'let ', 'const ', 
                      '#!/', 'SELECT ', 'FROM ', 'WHERE ', 'UPDATE ', 'INSERT ']
    code_count = sum(1 for indicator in code_indicators if indicator in text)
    if code_count > 2:  # Multiple code indicators suggest it's code
        return False
    
    # Check for excessive URLs
    url_count = len(re.findall(r'http[s]?://\S+', text))
    if url_count > 3:  # Too many URLs
        return False
    
    return True

def sample_wildchat_robust(target_samples=10, seed=42):
    """
    Create a robust sample of the WildChat dataset with language detection and random sampling.
    
    Args:
        target_samples: Number of samples to collect (default: 10)
        seed: Random seed for reproducibility
    """
    
    print(f"🚀 Starting robust WildChat sampling (target: {target_samples} samples)")
    print("🔍 Features: Language detection, random sampling, content quality checks")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Load the dataset
    print("📥 Loading WildChat dataset...")
    dataset = load_dataset("allenai/WildChat", split="train")
    print(f"✅ Loaded dataset with {len(dataset)} total entries")
    
    # Create shuffled indices for random sampling
    total_indices = list(range(len(dataset)))
    random.shuffle(total_indices)
    print(f"🎲 Shuffled dataset indices for random sampling")
    
    # Filter and collect valid entries
    filtered = []
    processed_count = 0
    language_rejected = 0
    length_rejected = 0
    content_rejected = 0
    conversation_rejected = 0
    
    print(f"🔄 Processing dataset (will stop at {target_samples} valid samples)...")
    
    for idx in tqdm(total_indices, desc="Sampling"):
        if len(filtered) >= target_samples:
            break
            
        row = dataset[idx]
        processed_count += 1
        
        # Check basic requirements
        if row.get("language") != "English":
            continue
            
        conversation = row.get("conversation", [])
        if len(conversation) < 2:
            conversation_rejected += 1
            continue
        
        question = conversation[0].get("content", "")
        answer = conversation[1].get("content", "")
        
        # Word count checks
        question_words = len(question.split())
        answer_words = len(answer.split())
        
        if not (question_words < 100 and 100 <= answer_words <= 300):
            length_rejected += 1
            continue
        
        # Language detection
        if not (detect_language(question) and detect_language(answer)):
            language_rejected += 1
            continue
        
        # Content quality checks
        if not (is_meaningful_content(question) and is_meaningful_content(answer)):
            content_rejected += 1
            continue
        
        # If we get here, it's a valid sample
        filtered.append({
            "model": row["model"],
            "question": question,
            "answer": answer,
            "original_index": idx,  # Track original position for debugging
            "question_word_count": question_words,
            "answer_word_count": answer_words
        })
        
        # Progress update every 100 valid samples (or for small targets, every sample)
        if len(filtered) % max(1, min(100, target_samples // 10)) == 0:
            print(f"✅ Found {len(filtered)}/{target_samples} valid samples")
    
    # Save to JSON
    output_filename = f"wildchat_{target_samples}_robust_sample.json"
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    
    # Calculate and print statistics
    print(f"\n{'='*60}")
    print(f"📊 SAMPLING RESULTS")
    print(f"{'='*60}")
    print(f"✅ Successfully saved {len(filtered)} entries to {output_filename}")
    print(f"📈 Processing efficiency: {len(filtered)}/{processed_count} ({len(filtered)/processed_count*100:.2f}%)")
    
    print(f"\n🚫 REJECTION BREAKDOWN:")
    print(f"   Language detection: {language_rejected}")
    print(f"   Length requirements: {length_rejected}")
    print(f"   Content quality: {content_rejected}")
    print(f"   Conversation structure: {conversation_rejected}")
    
    if filtered:
        print(f"\n📊 SAMPLE STATISTICS:")
        models = [entry['model'] for entry in filtered]
        unique_models = set(models)
        print(f"   Unique models: {len(unique_models)}")
        print(f"   Model distribution: {dict(Counter(models))}")
        
        avg_q_words = sum(entry['question_word_count'] for entry in filtered) / len(filtered)
        avg_a_words = sum(entry['answer_word_count'] for entry in filtered) / len(filtered)
        avg_q_chars = sum(len(entry['question']) for entry in filtered) / len(filtered)
        avg_a_chars = sum(len(entry['answer']) for entry in filtered) / len(filtered)
        
        print(f"   Average question length: {avg_q_words:.1f} words ({avg_q_chars:.0f} chars)")
        print(f"   Average answer length: {avg_a_words:.1f} words ({avg_a_chars:.0f} chars)")
        
        print(f"\n🔍 QUALITY CHECKS APPLIED:")
        print(f"   ✅ Language detection (filters non-English despite labels)")
        print(f"   ✅ Content meaningfulness (filters code/spam/repetitive text)")
        print(f"   ✅ Random sampling (not sequential)")
        print(f"   ✅ Reproducible (seed={seed})")
        
        # Show first few examples for verification
        print(f"\n📝 SAMPLE PREVIEW (first 2 examples):")
        for i, entry in enumerate(filtered[:2]):
            print(f"\n   Example {i+1} ({entry['model']}):")
            print(f"   Q: {entry['question'][:100]}...")
            print(f"   A: {entry['answer'][:100]}...")
    
    return filtered

def test_language_detection():
    """Test the language detection function with known examples."""
    
    test_cases = [
        ("Hello, how are you today?", True, "English"),
        ("Привет, как дела?", False, "Russian"),
        ("你好，你好吗？", False, "Chinese"),
        ("こんにちは、元気ですか？", False, "Japanese"),
        ("안녕하세요, 어떻게 지내세요?", False, "Korean"),
        ("مرحبا، كيف حالك؟", False, "Arabic"),
        ("This is a test with some émojis 😀", True, "English with emojis"),
        ("¡Hola! ¿Cómo estás? I speak English too.", True, "Mixed but mostly English"),
        ("def function(): return 'hello'", True, "Code but English"),
        ("", False, "Empty"),
    ]
    
    print("🧪 Testing language detection...")
    for text, expected, description in test_cases:
        result = detect_language(text)
        status = "✅" if result == expected else "❌"
        print(f"   {status} {description}: {result}")

if __name__ == "__main__":
    # Test language detection first
    test_language_detection()
    
    print()
    
    # Run the robust sampling
    sample_wildchat_robust(target_samples=10000, seed=42)
    
    print(f"\n🎉 Robust sampling complete!")
    print(f"💡 The sample is reproducible - run with same seed to get identical results")
    print(f"🔄 To get different samples, change the seed parameter")
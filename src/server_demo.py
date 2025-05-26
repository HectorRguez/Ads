import requests
import json
import time
from textwrap import fill
import difflib
from difflib import SequenceMatcher

# Server configuration
BASE_URL = 'http://localhost:8888'
HEADERS = {'Content-Type': 'application/json'}

def print_boxed_text(title, content, highlight_markers=None, width=78):
    """Helper function to print text in a box with optional highlighting."""
    border = "+" + "-" * (width) + "+"
    title_line = f"|{title.center(width)}|"
    
    lines = [border, title_line, border]
    
    # Wrap and process content
    wrapped_lines = fill(content, width=width-4).split('\n')
    
    for line in wrapped_lines:
        if highlight_markers:
            # Apply highlighting based on markers
            for start, end in highlight_markers.get(line, []):
                line = line[:start] + "**" + line[start:end] + "**" + line[end:]
        
        padded_line = f"| {line.ljust(width-2)} |"
        lines.append(padded_line)
    
    lines.append(border)
    return '\n'.join(lines)

def test_endpoint(name, method, endpoint, data=None):
    """Helper function to test endpoints with error handling"""
    print(f"\n{'='*80}")
    print(f"Testing: {name}")
    print(f"Endpoint: {method} {endpoint}")
    print(f"{'='*80}")
    
    try:
        response = requests.post(f"{BASE_URL}{endpoint}", headers=HEADERS, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Status Code: {response.status_code}")
            
            # Handle Q&A results with highlighting
            if 'inferred' in result:
                question = result.get('question', 'N/A')
                answer = result.get('inferred', 'N/A')
                
                # Show model info if available (for remote inference)
                if 'model' in result:
                    print(f"Model: {result['model']}")
                if 'usage' in result and result['usage']:
                    usage = result['usage']
                    print(f"Token Usage: {usage}")
                
                print("\nQ&A Generation Results:")
                print(print_boxed_text("QUESTION", question))
                print()
                print(print_boxed_text("ANSWER", answer))
            
            # Handle native ad insertion with highlighting
            elif 'modified_text' in result:                
                original_text = data.get('text', 'Original text not available') if data else 'Original text not available'
                modified_text = result['modified_text']
                
                print(print_boxed_text("ORIGINAL TEXT", original_text))
                print(print_boxed_text("MODIFIED TEXT", modified_text))

                if result['related_products']:
                    print(f"\nAll Related Products:")
                    for prod in result['related_products'][:3]:
                        print(f"   - {prod['name']} (similarity: {prod['similarity']:.3f})")
            
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            error_text = response.text
            print(f"Response: {error_text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def test_health_check():
    """Test the health endpoint"""
    print(f"\n{'='*80}")
    print("Testing: Health Check")
    print(f"{'='*80}")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            result = response.json()
            print("✅ Server is healthy!")
            print(f"Status: {result['status']}")
            print("Model Status:")
            for model, status in result['models'].items():
                status_icon = "✅" if status else "❌"
                print(f"  {status_icon} {model}: {status}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Health check exception: {e}")

def main():
    print("Starting Flask RAG Server Demo with Local and Remote Inference")
    
    # Health check first
    test_health_check()
    
    # Test questions for both local and remote inference
    test_questions = [
        {
            'question': 'What is machine learning?',
            'max_tokens': 1000
        }
    ]
    
    # Test both local and remote inference for each question
    for i, test_data in enumerate(test_questions, 1):
        print(f"\n{'#'*80}")
        print(f"INFERENCE TEST SET {i}")
        print(f"{'#'*80}")
        
        # Test local inference
        local_result = test_endpoint(
            f"Local Text Generation {i}", 
            "POST", 
            "/infer_local",
            test_data
        )
        
        # Add a small delay between requests
        time.sleep(1)
        
        # Test remote inference
        remote_result = test_endpoint(
            f"Remote Text Generation {i}", 
            "POST", 
            "/infer_remote",
            test_data
        )
    
    # Native Ad Insertion Tests  
    print(f"\n{'#'*80}")
    print("NATIVE AD INSERTION TESTS")
    print(f"{'#'*80}")
    
    ad_test_cases = [
        'Online privacy has become a major concern for internet users worldwide. With increasing surveillance, data collection, and cyber threats, many people are looking for ways to protect their digital footprint. There are several approaches to maintaining privacy online, including using secure browsers, enabling two-factor authentication, and being cautious about the information you share on social media platforms. Cybersecurity experts recommend using multiple layers of protection.',
        
        'The digital landscape continues to evolve rapidly, bringing both opportunities and challenges. As we become more dependent on technology for work, entertainment, and communication, it\'s crucial to understand the importance of cybersecurity. Malware, ransomware, and phishing attacks are becoming increasingly sophisticated, targeting both individuals and businesses. Advanced threat detection systems are essential for protection.'
    ]
    
    for i, test_case in enumerate(ad_test_cases, 1):
        test_endpoint(
            f"RAG Ad Insertion Test {i}", 
            "POST", 
            "/insert_native_ads",
            {
                'text': test_case,
                'max_tokens': 400
            }
        )
        time.sleep(1)

if __name__ == "__main__":
    main()
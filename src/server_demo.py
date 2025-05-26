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
            
            # Handle Q&A results (basic inference)
            if 'inferred' in result:
                question = result.get('question', 'N/A')
                answer = result.get('inferred', 'N/A')
                
                print("\nQ&A Generation Results:")
                print(print_boxed_text("QUESTION", question))
                print()
                print(print_boxed_text("ANSWER", answer))
            
            # Handle Q&A with native ads (combined inference)
            elif 'answer_with_ads' in result:
                question = result.get('question', 'N/A')
                original_answer = result.get('answer', 'N/A')
                answer_with_ads = result.get('answer_with_ads', 'N/A')
                
                print("\nQ&A with Native Ads Results:")
                print(print_boxed_text("QUESTION", question))
                print()
                print(print_boxed_text("ORIGINAL ANSWER", original_answer))
                print()
                print(print_boxed_text("ANSWER WITH ADS", answer_with_ads))
                
                if result.get('related_products'):
                    print(f"\nRelated Products Used:")
                    for prod in result['related_products'][:3]:
                        print(f"   - {prod['name']} (similarity: {prod['similarity']:.3f})")
            
            # Handle text with native ads (insert_native_ads)
            elif 'text_with_ads' in result:
                original_text = data.get('text', 'Original text not available') if data else 'Original text not available'
                text_with_ads = result['text_with_ads']
                
                print(print_boxed_text("ORIGINAL TEXT", original_text))
                print(print_boxed_text("TEXT WITH ADS", text_with_ads))

                if result.get('related_products'):
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
    print("Starting Flask RAG Server Demo with Native Advertising")
    
    # Health check first
    test_health_check()
    
    # Test native ad insertion with existing text
    print(f"\n{'#'*80}")
    print("NATIVE AD INSERTION TESTS (existing text)")
    print(f"{'#'*80}")
    
    ad_test_cases = [
        'Online privacy has become a major concern for internet users worldwide. With increasing surveillance, data collection, and cyber threats, many people are looking for ways to protect their digital footprint. There are several approaches to maintaining privacy online, including using secure browsers, enabling two-factor authentication, and being cautious about the information you share on social media platforms. Cybersecurity experts recommend using multiple layers of protection.',
        
        'The digital landscape continues to evolve rapidly, bringing both opportunities and challenges. As we become more dependent on technology for work, entertainment, and communication, it\'s crucial to understand the importance of cybersecurity. Malware, ransomware, and phishing attacks are becoming increasingly sophisticated, targeting both individuals and businesses. Advanced threat detection systems are essential for protection.'
    ]
    
    for i, test_case in enumerate(ad_test_cases, 1):
        test_endpoint(
            f"Text Ad Insertion Test {i}", 
            "POST", 
            "/insert_native_ads",
            {
                'text': test_case,
                'max_tokens': 400
            }
        )
        time.sleep(1)
    
    # Main feature: Combined Q&A with Native Ads (most representative)
    print(f"\n{'#'*80}")
    print("MAIN FEATURE: Q&A WITH NATIVE ADS")
    print(f"{'#'*80}")
    
    # Test questions for combined inference with ads
    test_questions_with_ads = [
        {
            'question': 'What is cybersecurity and why is it important?',
            'max_tokens': 500
        },
        {
            'question': 'How can businesses protect themselves from cyber threats?',
            'max_tokens': 500
        }
    ]
    
    # Test both local and remote inference with ads for each question
    for i, test_data in enumerate(test_questions_with_ads, 1):        
        # Test local inference with ads
        local_result = test_endpoint(
            f"Local Q&A with Native Ads {i}", 
            "POST", 
            "/infer_local_native_ads",
            test_data
        )
        
        time.sleep(2)
        
        # Test remote inference with ads
        remote_result = test_endpoint(
            f"Remote Q&A with Native Ads {i}", 
            "POST", 
            "/infer_remote_native_ads",
            test_data
        )
        
        time.sleep(2)
    
    print(f"\n{'='*80}")
    print("✅ Demo completed! The main feature showcases Q&A with integrated native ads.")
    print("This demonstrates how answers can be enhanced with relevant product recommendations.")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
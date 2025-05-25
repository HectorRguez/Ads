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

    try:
        response = requests.post(f"{BASE_URL}{endpoint}", headers=HEADERS, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Status Code: {response.status_code}")
            
            # Handle Q&A results with highlighting
            if 'inferred' in result:
                question = result.get('question', 'N/A')
                answer = result.get('inferred', 'N/A')
                
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
            print(f"Error: {response.status_code}")
            error_text = response.text
            print(f"Response: {error_text}")
            return None
            
    except Exception as e:
        print(f"Exception: {e}")
        return None

def main():
    print("Starting Flask RAG Server Demo with Native Advertising")

    # 1. Text Generation Test
    test_endpoint(
        "Text Generation (Inference)", 
        "POST", 
        "/infer_local",
        {
            'question': 'What is 2+2?',
            'max_tokens': 50
        }
    )
    
    # 2. Native Ad Insertion Tests  
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
            }
        )

if __name__ == "__main__":
    main()
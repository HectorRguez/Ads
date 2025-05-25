import requests
import json
import time

# Server configuration
BASE_URL = 'http://localhost:8888'
HEADERS = {'Content-Type': 'application/json'}

def print_separator():
    """Print a visual separator"""
    print("=" * 80)

def print_highlighted_box(title, content, width=80):
    """Print content in a highlighted box"""
    print("+" + "-" * (width - 2) + "+")
    print(f"| {title.center(width - 4)} |")
    print("+" + "-" * (width - 2) + "+")
    
    # Split content into lines and wrap if needed
    lines = content.split('\n')
    for line in lines:
        if len(line) <= width - 4:
            print(f"| {line.ljust(width - 4)} |")
        else:
            # Wrap long lines
            while len(line) > width - 4:
                print(f"| {line[:width-4].ljust(width - 4)} |")
                line = line[width-4:]
            if line:
                print(f"| {line.ljust(width - 4)} |")
    
    print("+" + "-" * (width - 2) + "+")

def test_endpoint(name, method, endpoint, data=None):
    """Helper function to test endpoints with error handling"""
    print_separator()
    print(f"Testing: {name}")
    print_separator()
    
    try:
        if method.upper() == 'GET':
            response = requests.get(f"{BASE_URL}{endpoint}")
        elif method.upper() == 'POST':
            response = requests.post(f"{BASE_URL}{endpoint}", headers=HEADERS, json=data)
        elif method.upper() == 'DELETE':
            response = requests.delete(f"{BASE_URL}{endpoint}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Status Code: {response.status_code}")
            
            # Handle Q&A results with highlighting
            if 'inferred' in result:
                question = result.get('question', 'N/A')
                answer = result.get('inferred', 'N/A')
                
                print("\nQ&A Generation Results:")
                print_highlighted_box("QUESTION", question)
                print()
                print_highlighted_box("ANSWER", answer)
            
            # Handle search results
            elif 'results' in result and len(result['results']) > 0:
                print(f"Search results ({len(result['results'])} found):")
                for i, item in enumerate(result['results'][:3]):
                    print(f"  {i+1}. {item.get('name', 'Unknown')} (Score: {item.get('similarity', 0):.3f})")
            
            # Handle native ad insertion with highlighting
            elif 'modified_text' in result:
                print("RAG-Powered Native Ad Insertion Results:")
                print(f"   Selected Product: {result.get('selected_product', {}).get('name', 'Unknown')}")
                print(f"   Similarity Score: {result.get('selected_product', {}).get('similarity_score', 0):.3f}")
                print(f"   Original length: {result['insertion_details']['original_length']} chars")
                print(f"   Modified length: {result['insertion_details']['modified_length']} chars")
                print(f"   Reasoning: {result['insertion_details']['reasoning']}")
                
                # Get original text from the request data
                original_text = data.get('text', 'Original text not available') if data else 'Original text not available'
                modified_text = result['modified_text']
                
                print()
                print_highlighted_box("ORIGINAL TEXT", original_text)
                print()
                print_highlighted_box("MODIFIED TEXT (WITH AD)", modified_text)
                
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
    
    # 2. Search Tests
    search_queries = [
        'privacy protection software'
    ]
    
    for query in search_queries:
        test_endpoint(
            f"Product Search: '{query}'", 
            "POST", 
            "/search",
            {
                'query': query,
                'top_k': 3
            }
        )
        time.sleep(0.5)
    
    # 3. Native Ad Insertion Tests  
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
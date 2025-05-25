import requests
import json
import time

# Server configuration
BASE_URL = 'http://localhost:8888'
HEADERS = {'Content-Type': 'application/json'}

def test_endpoint(name, method, endpoint, data=None):
    """Helper function to test endpoints with error handling"""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    try:
        if method.upper() == 'GET':
            response = requests.get(f"{BASE_URL}{endpoint}")
        elif method.upper() == 'POST':
            response = requests.post(f"{BASE_URL}{endpoint}", headers=HEADERS, json=data)
        elif method.upper() == 'DELETE':
            response = requests.delete(f"{BASE_URL}{endpoint}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success! Status Code: {response.status_code}")
            # Smart response formatting to avoid overwhelming output
            if 'results' in result and len(result['results']) > 0:
                print(f"Search results ({len(result['results'])} found):")
                for i, item in enumerate(result['results'][:3]):  # Show max 3 results
                    print(f"  {i+1}. {item.get('name', 'Unknown')} (Score: {item.get('similarity', 0):.3f})")
            elif 'inferred' in result:
                question = result.get('question', 'N/A')
                answer = result.get('inferred', 'N/A')
                print(f"Q&A Generation Results:")
                print(f"   Question: {question}")
                print(f"   Answer: {answer}")
            elif 'modified_text' in result:
                print(f"RAG-Powered Native Ad Insertion Results:")
                print(f"   Selected Product: {result.get('selected_product', {}).get('name', 'Unknown')}")
                print(f"   Similarity Score: {result.get('selected_product', {}).get('similarity_score', 0):.3f}")
                print(f"   Original length: {result['insertion_details']['original_length']} chars")
                print(f"   Modified length: {result['insertion_details']['modified_length']} chars")
                print(f"   Insertion point: {result['insertion_details']['insertion_point']}")
                print(f"   Reasoning: {result['insertion_details']['reasoning']}")
                print(f"   Related products found: {len(result['related_products'])}")
                print(f"\n Modified Text Preview:")
                modified_preview = result['modified_text'][:1000] + "..." if len(result['modified_text']) > 1000 else result['modified_text']
                print(f"   {modified_preview}")
                if result['related_products']:
                    print(f"\n All Related Products:")
                    for prod in result['related_products'][:3]:
                        print(f"   • {prod['name']} (similarity: {prod['similarity']:.3f})")
            else:
                # For other responses, limit output length
                response_str = json.dumps(result, indent=2)
                if len(response_str) > 200:
                    print(f"Response: {response_str[:200]}...")
                else:
                    print(f"Response: {response_str}")
            return result
        else:
            print(f"❌ Error: {response.status_code}")
            error_text = response.text
            if len(error_text) > 100:
                print(f"Response: {error_text[:100]}...")
            else:
                print(f"Response: {error_text}")
            return None
            
    except Exception as e:
        print(f"❌ Exception: {e}")
        return None

def main():
    print("Starting Flask RAG Server Demo with Native Advertising")

    # 1. Health Check
    test_endpoint("Health Check", "GET", "/health")
    
    # 2. Text Generation Test (using original parameters)
    test_endpoint(
        "Text Generation (Inference)", 
        "POST", 
        "/infer_local",
        {
            'question': 'What is 2+2?',
            'max_tokens': 50
        }
    )
    
    # 3. Search Tests
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
    
    # 4. Native Ad Insertion Tests (Main Feature!)    
    ad_test_cases = [
            ''''Online privacy has become a major concern for internet users worldwide. With increasing surveillance, data collection, and cyber threats, many people are looking for ways to protect their digital footprint. There are several approaches to maintaining privacy online, including using secure browsers, enabling two-factor authentication, and being cautious about the information you share on social media platforms. Cybersecurity experts recommend using multiple layers of protection.''',
        
            '''The digital landscape continues to evolve rapidly, bringing both opportunities and challenges. As we become more dependent on technology for work, entertainment, and communication, it's crucial to understand the importance of cybersecurity. Malware, ransomware, and phishing attacks are becoming increasingly sophisticated, targeting both individuals and businesses. Advanced threat detection systems are essential for protection.'''
        ,
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
import requests
import json
import time

# Server configuration
BASE_URL = 'http://localhost:8888'
HEADERS = {'Content-Type': 'application/json'}

def test_endpoint(name, method, endpoint, data=None):
    """Helper function to test endpoints with error handling"""
    print(f"\n{'='*60}")
    print(f"🧪 Testing: {name}")
    print(f"{'='*60}")
    
    try:
        if method.upper() == 'GET':
            response = requests.get(f"{BASE_URL}{endpoint}")
        elif method.upper() == 'POST':
            response = requests.post(f"{BASE_URL}{endpoint}", headers=HEADERS, json=data)
        elif method.upper() == 'DELETE':
            response = requests.delete(f"{BASE_URL}{endpoint}")
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Success!")
            
            # Smart response formatting to avoid overwhelming output
            if 'products' in result and len(result['products']) > 0:
                print(f"Found {len(result['products'])} products")
                if len(result['products']) <= 3:
                    for p in result['products']:
                        print(f"  • {p.get('name', 'Unknown')}")
                else:
                    print(f"  • {result['products'][0].get('name', 'Unknown')} (and {len(result['products'])-1} more)")
            elif 'results' in result and len(result['results']) > 0:
                print(f"Search results ({len(result['results'])} found):")
                for i, item in enumerate(result['results'][:3]):  # Show max 3 results
                    print(f"  {i+1}. {item.get('name', 'Unknown')} (Score: {item.get('similarity', 0):.3f})")
            elif 'embedding' in result:
                print(f"Generated embedding (dimension: {len(result['embedding'])})")
            elif 'embeddings' in result:
                print(f"Generated {result.get('count', 0)} embeddings")
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
    print("🚀 Starting Flask RAG Server Demo")
    print("Make sure your server is running on localhost:8888")
    
    # 1. Health Check
    test_endpoint("Health Check", "GET", "/health")
    
    # 2. Original Inference Test
    test_endpoint(
        "Text Generation (Inference)", 
        "POST", 
        "/infer",
        {'prompt': 'You are a helpful assistant. \nQuestion:\nWhat is 2+2?.\nAnswer:\n'}
    )
    
    # 3. Original Ad Insertion Test
    test_endpoint(
        "Native Ad Insertion", 
        "POST", 
        "/insert_native_ads",
        {'text': 'This is some sample text content.'}
    )
    
    # 4. Embedding Generation
    test_endpoint(
        "Single Text Embedding", 
        "POST", 
        "/embed",
        {
            'text': 'VPN service with strong encryption',
            'instruction': 'Represent this sentence for searching relevant passages:'
        }
    )
    
    # 5. Batch Embedding Generation
    test_endpoint(
        "Batch Text Embeddings", 
        "POST", 
        "/embed_batch",
        {
            'texts': [
                'VPN with strong security',
                'Fast internet privacy tool',
                'Secure browsing software'
            ],
            'instruction': 'Represent this sentence for searching relevant passages:'
        }
    )

    # 6. Add a New Product
    new_product_result = test_endpoint(
        "Add New Product", 
        "POST", 
        "/products",
        {
            'name': 'ProtonVPN',
            'category': 'VPN/Privacy',
            'description': 'Swiss-based VPN service with end-to-end encryption, no-logs policy, and Secure Core servers for maximum privacy protection.'
        }
    )
    
    # 7. Add Duplicate Product (should fail/skip)
    test_endpoint(
        "Add Duplicate Product (should skip)", 
        "POST", 
        "/products",
        {
            'name': 'ProtonVPN',  # Same name as above
            'category': 'VPN/Privacy',
            'description': 'Duplicate entry test'
        }
    )
    
    # 9. Search Tests
    search_queries = [
        'VPN with strong security features',
        'privacy protection software',
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
        time.sleep(0.5)  # Small delay between searches
    
if __name__ == "__main__":
    main()
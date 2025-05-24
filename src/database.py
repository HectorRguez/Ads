import sqlite3
import numpy as np
import json
import requests
from typing import List, Tuple, Dict

class SimpleProductRAG:
    def __init__(self, db_path: str = "products.db", mistral_url: str = "http://localhost:8888"):
        self.db_path = db_path
        self.mistral_url = mistral_url
        self.setup_database()
    
    def setup_database(self):
        """Create simple products table"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding from Mistral server"""
        try:
            response = requests.post(
                f"{self.mistral_url}/embed",
                json={"text": text},
                headers={"Content-Type": "application/json"},
                timeout=30  # Add timeout for better error handling
            )
            
            if response.status_code == 200:
                result = response.json()
                embedding = np.array(result["embedding"], dtype=np.float32)
                
                # Normalize the embedding for cosine similarity
                embedding = embedding / np.linalg.norm(embedding)
                
                return embedding
            else:
                print(f"Embedding API error: {response.status_code} - {response.text}")
                raise Exception(f"Embedding API failed: {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        except Exception as e:
            print(f"Embedding error: {e}")
        
    def add_product(self, name: str, category: str, description: str):
        """Add a product to the database"""
        # Combine product info for embedding
        full_text = f"Name: {name}\nCategory: {category}\nDescription: {description}"
        
        # Get embedding
        embedding = self.get_embedding(full_text)
        embedding_bytes = embedding.tobytes()
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO products (name, category, description, embedding)
            VALUES (?, ?, ?, ?)
        """, (name, category, description, embedding_bytes))
        conn.commit()
        conn.close()
        
        print(f"Added product: {name}")
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, str, float]]:
        """Search for products similar to the query"""
        # Get query embedding
        query_embedding = self.get_embedding(query)
        
        # Get all products from database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT name, category, description, embedding FROM products")
        
        results = []
        for name, category, description, embedding_bytes in cursor:
            # Convert bytes back to numpy array
            product_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            # Calculate similarity
            similarity = self.cosine_similarity(query_embedding, product_embedding)
            results.append((name, category, description, similarity))
        
        conn.close()
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x[3], reverse=True)
        return results[:top_k]
    
    def get_all_products(self) -> List[Dict]:
        """Get all products from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT id, name, category, description FROM products")
        
        products = []
        for row in cursor:
            products.append({
                "id": row[0],
                "name": row[1], 
                "category": row[2],
                "description": row[3]
            })
        
        conn.close()
        return products
    
    def delete_product(self, product_id: int):
        """Delete a product by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM products WHERE id = ?", (product_id,))
        conn.commit()
        conn.close()
        print(f"Deleted product with ID: {product_id}")

# Example usage
if __name__ == "__main__":
    # Initialize the RAG system
    rag = SimpleProductRAG()
    
    # Add some sample products
    sample_products = [
        ("iPhone 15", "Electronics", "Latest Apple smartphone with advanced camera system"),
        ("Nike Air Max", "Footwear", "Comfortable running shoes with air cushioning technology"),
        ("MacBook Pro", "Electronics", "Powerful laptop for professionals with M3 chip"),
        ("Levi's Jeans", "Clothing", "Classic denim jeans with comfortable fit"),
        ("Sony Headphones", "Electronics", "Noise-canceling wireless headphones with premium sound")
    ]
    
    print("Adding products...")
    for name, category, description in sample_products:
        rag.add_product(name, category, description)
    
    print("\nProducts added successfully!")
    
    # Search examples
    queries = [
        "smartphone with good camera",
        "comfortable running gear", 
        "laptop for work",
        "wireless audio device"
    ]
    
    print("\nSearch Results:")
    print("=" * 50)
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        print("-" * 30)
        
        results = rag.search(query, top_k=3)
        for name, category, description, score in results:
            print(f"Score: {score:.3f} | {name} ({category})")
            print(f"  Description: {description[:60]}...")
            print()
    
    # Show all products
    print("\nAll products in database:")
    all_products = rag.get_all_products()
    for product in all_products:
        print(f"ID: {product['id']} | {product['name']} ({product['category']})")
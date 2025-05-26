import os
import csv
import sqlite3
import numpy as np
from typing import List, Tuple, Dict

class RAGSystem:
    def __init__(self, embedding_model, db_path="products.db"):
        self.embedding_model = embedding_model
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Create products table with URL field"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                url TEXT,
                description TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def generate_embedding(self, text):
        """Generate embedding for text"""
        instruction = "Represent this sentence for searching relevant passages:"
        formatted_text = f"Instruct: {instruction}\nQuery: {text}"
        
        embedding = self.embedding_model.encode(
            formatted_text,
            normalize_embeddings=True,
            convert_to_tensor=False,
            show_progress_bar=False
        )
        return embedding
    
    def add_product(self, name, category, url, description):
        """Add product to database"""
        # Check if exists
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM products WHERE name = ?", (name,))
        if cursor.fetchone()[0] > 0:
            conn.close()
            print(f"Product '{name}' already exists")
            return False
        
        # Generate embedding (exclude URL from embedding)
        full_text = f"Name: {name}\nCategory: {category}\nDescription: {description}"
        embedding = self.generate_embedding(full_text)
        embedding_bytes = embedding.tobytes()
        
        # Insert to database
        try:
            conn.execute("""
                INSERT INTO products (name, category, url, description, embedding)
                VALUES (?, ?, ?, ?, ?)
            """, (name, category, url, description, embedding_bytes))
            conn.commit()
            print(f"Added product: {name}")
            return True
        except sqlite3.IntegrityError:
            print(f"Product '{name}' already exists")
            return False
        finally:
            conn.close()
    
    def search_products(self, query, top_k=5):
        """Search for similar products and return URL in results"""
        query_embedding = self.generate_embedding(query)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT name, category, url, description, embedding FROM products")
        
        results = []
        for name, category, url, description, embedding_bytes in cursor:
            product_embedding = np.frombuffer(embedding_bytes, dtype=np.float16)
            similarity = np.dot(query_embedding, product_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(product_embedding)
            )
            results.append((name, category, url, description, similarity))
        
        conn.close()
        results.sort(key=lambda x: x[4], reverse=True)  # Sort by similarity (index 4)
        return results[:top_k]
    
    def get_all_products(self):
        """Get all products including URL"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT id, name, category, url, description FROM products")
        products = [{"id": row[0], "name": row[1], "category": row[2], "url": row[3], "description": row[4]} 
                   for row in cursor]
        conn.close()
        return products
    
    def delete_product(self, product_id):
        """Delete product by ID"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("DELETE FROM products WHERE id = ?", (product_id,))
        conn.commit()
        conn.close()
    
    def load_from_csv(self, csv_path):
        """Load products from CSV file with URL support"""
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return
        
        added_count = 0
        with open(csv_path, 'r', encoding='utf-8') as file:

            delimiter = ','
            reader = csv.DictReader(file, delimiter=delimiter)
            for row in reader:
                name = row.get('Name', '').strip()
                category = row.get('Category', '').strip()
                url = row.get('URL', '').strip()
                description = row.get('Description', '').strip()
                
                if name and category and description:
                    if self.add_product(name, category, url, description):
                        added_count += 1
        
        print(f"Loaded {added_count} products from CSV")

def init_rag_system(config, embedding_model):
    """Initialize RAG system and load data"""
    rag_system = RAGSystem(embedding_model)
    
    # Load CSV data if available
    csv_path = config.get('data', 'csv_path', fallback='products.csv')
    if os.path.exists(csv_path):
        rag_system.load_from_csv(csv_path)
    
    print("✅ RAG system initialized")
    return rag_system
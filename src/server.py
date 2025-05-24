import os
import sys
import csv
import configparser
import sqlite3
import numpy as np
from typing import List, Tuple, Dict
from flask import Flask, request, jsonify
from flasgger import Swagger
from llama_cpp import Llama, llama_cpp
from sentence_transformers import SentenceTransformer
import torch

app = Flask(__name__)
swagger = Swagger(app)

hostname = '0.0.0.0'
port = 8888

model = None  # For text generation (Mistral)
embedding_model = None  # For embeddings (Stella)

class ProductRAG:
    def __init__(self, db_path: str = "products.db"):
        self.db_path = db_path
        self.setup_database()
    
    def setup_database(self):
        """Create simple products table"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding using the loaded embedding model"""
        if embedding_model is None:
            raise Exception("Embedding model not loaded")
        
        try:
            # Format text for Stella model
            instruction = "Represent this sentence for searching relevant passages:"
            formatted_text = f"Instruct: {instruction}\nQuery: {text}"
            
            # Generate embedding
            embedding = embedding_model.encode(
                formatted_text,
                normalize_embeddings=True,
                convert_to_tensor=False,
                show_progress_bar=False
            )
            
            return embedding
            
        except Exception as e:
            print(f"Embedding error: {e}")
            raise
    
    def product_exists(self, name: str) -> bool:
        """Check if a product already exists in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("SELECT COUNT(*) FROM products WHERE name = ?", (name,))
        count = cursor.fetchone()[0]
        conn.close()
        return count > 0
    
    def add_product(self, name: str, category: str, description: str):
        """Add a product to the database"""
        if self.product_exists(name):
            print(f"Product '{name}' already exists, skipping...")
            return False
        
        # Combine product info for embedding
        full_text = f"Name: {name}\nCategory: {category}\nDescription: {description}"
        
        # Get embedding
        embedding = self.get_embedding(full_text)
        embedding_bytes = embedding.tobytes()
        
        # Store in database
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                INSERT INTO products (name, category, description, embedding)
                VALUES (?, ?, ?, ?)
            """, (name, category, description, embedding_bytes))
            conn.commit()
            print(f"Added product: {name}")
            return True
        except sqlite3.IntegrityError:
            print(f"Product '{name}' already exists (integrity constraint)")
            return False
        finally:
            conn.close()
    
    def load_from_csv(self, csv_path: str):
        """Load products from CSV file"""
        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return
        
        added_count = 0
        skipped_count = 0
        
        print(f"Loading products from {csv_path}...")
        
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter='\t')
            
            for row in reader:
                name = row.get('Name', '').strip()
                category = row.get('Category', '').strip()
                description = row.get('Description', '').strip()
                
                if name and category and description:
                    if self.add_product(name, category, description):
                        added_count += 1
                    else:
                        skipped_count += 1
                else:
                    print(f"Skipping incomplete row: {row}")
        
        print(f"CSV loading complete: {added_count} added, {skipped_count} skipped")
    
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

# Initialize RAG system
rag_system = None

@app.route('/search', methods=['POST'])
def search_endpoint():
    """
    Search for products using RAG.
    ---
    parameters:
      - name: query
        in: body
        required: true
        schema:
          type: object
          properties:
            query:
              type: string
              description: Search query
            top_k:
              type: integer
              description: Number of results to return (default 5)
    responses:
      200:
        description: Search results
        schema:
          type: object
          properties:
            results:
              type: array
              items:
                type: object
                properties:
                  name:
                    type: string
                  category:
                    type: string
                  description:
                    type: string
                  similarity:
                    type: number
    """
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    query = data.get('query')
    top_k = data.get('top_k', 5)

    if not query:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    try:
        results = rag_system.search(query, top_k)
        
        formatted_results = []
        for name, category, description, similarity in results:
            formatted_results.append({
                "name": name,
                "category": category,
                "description": description,
                "similarity": float(similarity)
            })
        
        return jsonify({"results": formatted_results})
        
    except Exception as e:
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@app.route('/products', methods=['GET'])
def get_products_endpoint():
    """
    Get all products from the database.
    ---
    responses:
      200:
        description: List of all products
        schema:
          type: object
          properties:
            products:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: integer
                  name:
                    type: string
                  category:
                    type: string
                  description:
                    type: string
    """
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"}), 500

    try:
        products = rag_system.get_all_products()
        return jsonify({"products": products})
    except Exception as e:
        return jsonify({"error": f"Failed to get products: {str(e)}"}), 500

@app.route('/products', methods=['POST'])
def add_product_endpoint():
    """
    Add a new product to the database.
    ---
    parameters:
      - name: product
        in: body
        required: true
        schema:
          type: object
          properties:
            name:
              type: string
              description: Product name
            category:
              type: string
              description: Product category
            description:
              type: string
              description: Product description
    responses:
      200:
        description: Product added successfully
      400:
        description: Bad request
    """
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    name = data.get('name')
    category = data.get('category')
    description = data.get('description')

    if not all([name, category, description]):
        return jsonify({"error": "Missing required fields: name, category, description"}), 400

    try:
        success = rag_system.add_product(name, category, description)
        if success:
            return jsonify({"message": f"Product '{name}' added successfully"})
        else:
            return jsonify({"message": f"Product '{name}' already exists"}), 409
    except Exception as e:
        return jsonify({"error": f"Failed to add product: {str(e)}"}), 500

@app.route('/products/<int:product_id>', methods=['DELETE'])
def delete_product_endpoint(product_id):
    """
    Delete a product by ID.
    ---
    parameters:
      - name: product_id
        in: path
        type: integer
        required: true
        description: Product ID to delete
    responses:
      200:
        description: Product deleted successfully
      500:
        description: Error deleting product
    """
    if rag_system is None:
        return jsonify({"error": "RAG system not initialized"}), 500

    try:
        rag_system.delete_product(product_id)
        return jsonify({"message": f"Product with ID {product_id} deleted successfully"})
    except Exception as e:
        return jsonify({"error": f"Failed to delete product: {str(e)}"}), 500

@app.route('/infer', methods=['POST'])
def infer_endpoint():
    """
    Perform inference on a prompt.
    ---
    parameters:
      - name: prompt
        in: body
        required: true
        schema:
          type: object
          properties:
            prompt:
              type: string
              description: The input prompt for inference
    responses:
      200:
        description: Successful inference
        schema:
          type: object
          properties:
            inferred:
              type: string
              description: The inferred result
      400:
        description: Bad Request - Missing 'prompt'
      415:
        description: Unsupported Media Type - Request must be JSON
      500:
        description: Internal Server Error - Model not loaded
      500:
        description: Inference Error - An error occurred during inference
    """
    if model is None:
        return jsonify({"error": "Server is not fully initialized (model not loaded)"}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    try:
        inference_result = model(
            prompt,
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.2,
            stop=["\n", "\n\n", "</s>", "Question:", "?"]
        )

        if inference_result and inference_result['choices']:
            inferred_result = inference_result['choices'][0]['text'].strip()
            return jsonify({"inferred": inferred_result})
        else:
            return jsonify({"error": "Inference failed; check server logs"}), 500

    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

@app.route('/embed', methods=['POST'])
def embed_endpoint():
    """
    Generate embeddings using Stella-en-1.5B model.
    ---
    parameters:
      - name: text
        in: body
        required: true
        schema:
          type: object
          properties:
            text:
              type: string
              description: The input text to embed
            instruction:
              type: string
              description: Optional instruction for the embedding (default: query)
    responses:
      200:
        description: Successful embedding generation
        schema:
          type: object
          properties:
            embedding:
              type: array
              items:
                type: number
              description: The embedding vector
            dimension:
              type: integer
              description: The dimension of the embedding
            model:
              type: string
              description: The model used for embedding
      400:
        description: Bad Request - Missing 'text'
      415:
        description: Unsupported Media Type - Request must be JSON
      500:
        description: Internal Server Error - Model not loaded
    """
    if embedding_model is None:
        return jsonify({"error": "Embedding model not loaded"}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    text = data.get('text')
    instruction = data.get('instruction', 'Represent this sentence for searching relevant passages:')

    if not text:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    try:
        # Stella uses instruction-based embeddings
        # Format: "Instruct: {instruction}\nQuery: {text}"
        formatted_text = f"Instruct: {instruction}\nQuery: {text}"
        
        print(f"[DEBUG] Generating embedding for: {formatted_text[:100]}...")
        
        # Generate embedding
        embedding = embedding_model.encode(
            formatted_text,
            normalize_embeddings=True,  # Normalize for cosine similarity
            convert_to_tensor=False,    # Return numpy array
            show_progress_bar=False
        )
        
        # Convert to list for JSON serialization
        embedding_list = embedding.tolist()
        
        print(f"[DEBUG] Generated embedding with dimension: {len(embedding_list)}")
        
        return jsonify({
            "embedding": embedding_list,
            "dimension": len(embedding_list),
            "instruction": instruction
        })
        
    except Exception as e:
        print(f"[ERROR] Embedding generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Embedding generation failed: {str(e)}"}), 500

@app.route('/embed_batch', methods=['POST'])
def embed_batch_endpoint():
    """Generate embeddings for multiple texts (batch processing)."""
    if embedding_model is None:
        return jsonify({"error": "Embedding model not loaded"}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    texts = data.get('texts')
    instruction = data.get('instruction', 'Represent this sentence for searching relevant passages:')

    if not texts or not isinstance(texts, list):
        return jsonify({"error": "Missing 'texts' array in request body"}), 400

    try:
        # Format all texts with instruction
        formatted_texts = [f"Instruct: {instruction}\nQuery: {text}" for text in texts]
        
        print(f"[DEBUG] Generating {len(formatted_texts)} embeddings...")
        
        # Generate embeddings in batch
        embeddings = embedding_model.encode(
            formatted_texts,
            normalize_embeddings=True,
            convert_to_tensor=False,
            show_progress_bar=False,
            batch_size=32  # Process in batches of 32
        )
        
        # Convert to list for JSON serialization
        embeddings_list = embeddings.tolist()
        
        print(f"[DEBUG] Generated {len(embeddings_list)} embeddings")
        
        return jsonify({
            "embeddings": embeddings_list,
            "dimension": len(embeddings_list[0]) if embeddings_list else 0,
            "count": len(embeddings_list),
            "model": "stella-en-1.5B-v5"
        })
        
    except Exception as e:
        print(f"[ERROR] Batch embedding error: {str(e)}")
        return jsonify({"error": f"Batch embedding failed: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_endpoint():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "models": {
            "text_generation": model is not None,
            "embeddings": embedding_model is not None,
            "rag_system": rag_system is not None
        }
    })

@app.route('/insert_native_ads', methods=['POST'])
def insert_native_ads_endpoint():
    """Insert native ads into text."""
    if model is None:
         return jsonify({"error": "Server is not fully initialized (model not loaded)"}), 500

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415

    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({"error": "Missing 'text' in request body"}), 400

    print(f"Received text for ad insertion: {text}")
    text_with_ad_result = f"Text with ad inserted: {text} [AD: Buy our amazing product!]"

    return jsonify({"text_with_ad": text_with_ad_result})


if __name__ == '__main__':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.ini')

    config = configparser.ConfigParser()

    if not os.path.exists(config_path):
        print(f"Error: config.ini not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    try:
        config.read(config_path)

        hostname = config.get('server', 'hostname', fallback='0.0.0.0')
        port = config.getint('server', 'port', fallback=8888)

        # Text generation model (Mistral)
        model_path = config.get('model', 'path')
        max_tokens = config.getint('model', 'max_tokens', fallback=2048)
        
        # Embedding model (Stella)
        embedding_model_path = config.get('embedding', 'path', fallback='./models/stella-en-1.5B')
        
        # CSV file path (optional)
        csv_file_path = config.get('data', 'csv_path', fallback='products.csv')
        
    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"Error reading config.ini: {e}", file=sys.stderr)
        print("Please ensure config.ini has '[server]', '[model]' and '[embedding]' sections with required options.", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
         print(f"Error parsing config.ini: {e}", file=sys.stderr)
         print("Please ensure port is an integer.", file=sys.stderr)
         sys.exit(1)

    
    # Load text generation model (Mistral)
    try:
        model_path = config.get('model', 'path')
        model_gpu_device = config.getint('model', 'gpu_device', fallback=0)
        
        print(f"Loading text generation model from: {model_path}")
        print(f"Using GPU device {model_gpu_device} for text generation model")
        
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=config.getint('model', 'max_tokens', fallback=2048),
            split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,  # Use single GPU only
            main_gpu=model_gpu_device,
            verbose=False
        )
        print(f"✅ Text generation model loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading text generation model from {model_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Load embedding model (Stella)
    try:
        embedding_model_path = config.get('embedding', 'path')
        embedding_gpu_device = config.getint('embedding', 'gpu_device', fallback=0)
        
        print(f"Loading embedding model from: {embedding_model_path}")
        
        # Set device for embedding model
        if torch.cuda.is_available():
            device = f'cuda:{embedding_gpu_device}'
        else:
            device = 'cpu'
        
        print(f"Using device for embeddings: {device}")
        
        embedding_model = SentenceTransformer(
            embedding_model_path,
            device=device,
            trust_remote_code=True
        )
        
        # Test the model with a sample text
        test_embedding = embedding_model.encode("test", show_progress_bar=False)
        print(f"✅ Embedding model loaded successfully (dimension: {len(test_embedding)})")

    except Exception as e:
        print(f"❌ Error loading embedding model from {embedding_model_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Initialize RAG system
    try:
        print("Initializing RAG system...")
        rag_system = ProductRAG()
        
        # Load CSV data if file exists
        if os.path.exists(csv_file_path):
            print(f"Loading data from CSV: {csv_file_path}")
            rag_system.load_from_csv(csv_file_path)
        else:
            print(f"CSV file not found: {csv_file_path} - skipping CSV import")
        
        print("✅ RAG system initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}", file=sys.stderr)
        sys.exit(1)

    # Optional: Print final GPU memory usage summary
    if torch.cuda.is_available():
        print("\n=== GPU Memory Summary ===")
        for i in range(torch.cuda.device_count()):
            memory_allocated = torch.cuda.memory_allocated(i) / 1024**3
            memory_reserved = torch.cuda.memory_reserved(i) / 1024**3
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {memory_allocated:.2f}GB/{total_memory:.2f}GB allocated")

    print(f"🚀 Starting Flask server on {hostname}:{port}")
    print(f"📝 Text generation: Mistral 7B")
    print(f"🔍 Embeddings: Stella-en-1.5B")
    print(f"🗄️  RAG Database: SQLite with {len(rag_system.get_all_products())} products")
    print(f"📖 API docs: http://{hostname}:{port}/apidocs/")
    
    app.run(host=hostname, port=port, debug=False)
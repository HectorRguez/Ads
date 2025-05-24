import os
import sys
import configparser
from flask import Flask, request, jsonify
from flasgger import Swagger
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

app = Flask(__name__)

swagger = Swagger(app)

hostname = '0.0.0.0'
port = 8888

model = None  # For text generation (Mistral)
embedding_model = None  # For embeddings (Stella)

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
            "model": "stella-en-1.5B-v5",
            "instruction": instruction
        })
        
    except Exception as e:
        print(f"[ERROR] Embedding generation error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Embedding generation failed: {str(e)}"}), 500

@app.route('/embed_batch', methods=['POST'])
def embed_batch_endpoint():
    """
    Generate embeddings for multiple texts (batch processing).
    ---
    parameters:
      - name: texts
        in: body
        required: true
        schema:
          type: object
          properties:
            texts:
              type: array
              items:
                type: string
              description: Array of texts to embed
            instruction:
              type: string
              description: Optional instruction for the embeddings
    responses:
      200:
        description: Successful batch embedding generation
        schema:
          type: object
          properties:
            embeddings:
              type: array
              items:
                type: array
                items:
                  type: number
              description: Array of embedding vectors
            dimension:
              type: integer
              description: The dimension of each embedding
            count:
              type: integer
              description: Number of embeddings generated
    """
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
            "embeddings": embedding_model is not None
        }
    })

@app.route('/insert_native_ads', methods=['POST'])
def insert_native_ads_endpoint():
    """
    Insert native ads into text.
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
              description: The input text for ad insertion
    responses:
      200:
        description: Successful ad insertion
        schema:
          type: object
          properties:
            text_with_ad:
              type: string
              description: The text with ads inserted
      400:
        description: Bad Request - Missing 'text'
      415:
        description: Unsupported Media Type - Request must be JSON
      500:
        description: Internal Server Error - Model not loaded
    """
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
        print(f"Loading text generation model from: {model_path}")
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            n_ctx=2048,
            verbose=False
        )
        print(f"✅ Text generation model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading text generation model from {model_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Load embedding model (Stella)
    try:
        print(f"Loading embedding model from: {embedding_model_path}")
        
        # Check if we have GPU available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device for embeddings: {device}")
        
        embedding_model = SentenceTransformer(
            embedding_model_path,
            device=device,
            trust_remote_code=True  # Needed for some models
        )
        
        # Test the model with a sample text
        test_embedding = embedding_model.encode("test", show_progress_bar=False)
        print(f"✅ Embedding model loaded successfully (dimension: {len(test_embedding)})")
        
    except Exception as e:
        print(f"❌ Error loading embedding model from {embedding_model_path}: {e}", file=sys.stderr)
        print("Make sure you have downloaded the Stella model to the specified path.")
        sys.exit(1)

    print(f"🚀 Starting Flask server on {hostname}:{port}")
    print(f"📝 Text generation: Mistral 7B")
    print(f"🔍 Embeddings: Stella-en-1.5B")
    print(f"📖 API docs: http://{hostname}:{port}/apidocs/")
    
    app.run(host=hostname, port=port, debug=False)
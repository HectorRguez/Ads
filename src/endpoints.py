from flask import request, jsonify
from models import generate_text
from datetime import datetime
import os
import requests

def load_template(config, template_key):
    """Load a template file from config"""
    template_path = config.get('prompts', template_key)
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()

def generate_text_local(text_model, prompt, max_tokens):
    """Generate text using local model"""
    return generate_text(text_model, prompt, max_tokens)

def generate_text_remote(prompt, max_tokens, model='deepseek-chat', temperature=0.7):
    """Generate text using DeepSeek API"""
    api_key = os.getenv('DEEPSEEK_API_KEY')
    if not api_key:
        raise ValueError("DeepSeek API key not configured")
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    payload = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': prompt
            }
        ],
        'max_tokens': max_tokens,
        'temperature': temperature
    }
    
    response = requests.post(
        'https://api.deepseek.com/v1/chat/completions',
        headers=headers,
        json=payload,
        timeout=30
    )
    
    if response.status_code != 200:
        raise requests.exceptions.RequestException(f"DeepSeek API error: {response.status_code} - {response.text}")
    
    api_response = response.json()
    
    if 'choices' in api_response and len(api_response['choices']) > 0:
        result = api_response['choices'][0]['message']['content']
        usage = api_response.get('usage', {})
        return result, usage
    else:
        raise ValueError("Unexpected API response format")

def get_related_products_and_ad(rag_system, config, prompt, text_content, max_tokens, text_generator_func):
    """
    Common function to get related products and generate native ad insertion
    
    Args:
        rag_system: RAG system instance
        config: Configuration object
        prompt: Text prompt that was used to generate text_content
        text_content: Text content to analyze for product matching
        max_tokens: Maximum tokens for ad generation
        text_generator_func: Function to generate text (local or remote)
    
    Returns:
        tuple: (generated_ad_text, selected_product, related_products, usage_info)
    """
    # Step 1: Search for related products
    related_products = rag_system.search_products(text_content, top_k=3)
    
    if not related_products:
        raise ValueError("No products found in database for ad insertion")
    
    # Step 2: Select the best matching product
    selected_product = related_products[0]
    company_name = selected_product[0]
    category = selected_product[1]
    description = selected_product[2]
    similarity_score = selected_product[3]

    # TODO: Add the product link retrieval logic here
    product_link = None
    
    # Step 3: Load ad insertion template
    if product_link:
        ad_template_path = config.get('prompts', 'ad_with_url_insertion_template_path')
        with open(ad_template_path, 'r', encoding='utf-8') as f:
            ad_template = f.read()
    else:
        ad_template_path = config.get('prompts', 'ad_without_url_insertion_template_path')
        with open(ad_template_path, 'r', encoding='utf-8') as f:
            ad_template = f.read()
    
    # Step 4: Create prompt for ad insertion
    ad_prompt = ad_template.format(
        original_prompt=prompt,
        original_text=text_content,
        company_name=company_name,
        product_link=product_link,
        category=category,
        description=description
    )
    
    # Step 5: Generate ad text
    generated_response, usage_info = text_generator_func(ad_prompt, max_tokens)
    
    # Step 6: Format results
    selected_product_info = {
        "name": company_name,
        "category": category,
        "description": description,
        "similarity_score": float(similarity_score)
    }
    
    related_products_info = [{
        "name": name,
        "category": cat,
        "description": desc,
        "similarity": float(sim)
    } for name, cat, desc, sim in related_products]
    
    return generated_response, selected_product_info, related_products_info, usage_info

def register_endpoints(app, text_model, embedding_model, rag_system, config):
    """Register all API endpoints"""
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check"""
        return jsonify({
            "status": "healthy",
            "models": {
                "text_generation": text_model is not None,
                "embeddings": embedding_model is not None,
                "rag_system": rag_system is not None
            }
        })
    
    @app.route('/infer_local', methods=['POST'])
    def infer_local():
        """Text generation endpoint using local model"""
        data = request.get_json()
        question = data.get('question')
        max_tokens = data.get('max_tokens', 100)
        
        if not question:
            return jsonify({"error": "Missing 'question'"}), 400
        
        try:
            question_template = load_template(config, 'qa_template_path')
            question_prompt = question_template.format(question=question)
            result = generate_text_local(text_model, question_prompt, max_tokens)
            
            return jsonify({
                "question": question,
                "prompt": question_prompt,
                "inferred": result,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/infer_remote', methods=['POST'])
    def infer_remote():
        """Text generation endpoint using DeepSeek API"""
        data = request.get_json()
        question = data.get('question')
        max_tokens = data.get('max_tokens', 100)
        model = data.get('model', 'deepseek-chat')
        temperature = data.get('temperature', 0.7)
        
        if not question:
            return jsonify({"error": "Missing 'question'"}), 400
        
        try:
            question_template = load_template(config, 'qa_template_path')
            question_prompt = question_template.format(question=question)
            result, usage = generate_text_remote(question_prompt, max_tokens, model, temperature)
            
            return jsonify({
                "question": question,
                "prompt": question_prompt,
                "inferred": result,
                "model": model,
                "usage": usage
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/infer_local_native_ads', methods=['POST'])
    def infer_local_native_ads():
        """Text generation with native ads using local model"""
        data = request.get_json()
        question = data.get('question')
        max_tokens = data.get('max_tokens', 100)
        ad_max_tokens = data.get('ad_max_tokens', 300)
        
        if not question:
            return jsonify({"error": "Missing 'question'"}), 400
        
        try:
            # Step 1: Generate answer to question
            question_template = load_template(config, 'qa_template_path')
            question_prompt = question_template.format(question=question)
            answer = generate_text_local(text_model, question_prompt, max_tokens)
            
            # Step 2: Generate native ads based on the answer
            def local_text_generator(prompt, tokens):
                return generate_text_local(text_model, prompt, tokens), {}
            
            ad_text, selected_product, related_products, _ = get_related_products_and_ad(
                rag_system, config, answer, ad_max_tokens, local_text_generator
            )
            
            return jsonify({
                "question": question,
                "answer": answer,
                "answer_with_ads": ad_text,
                "selected_product": selected_product,
                "related_products": related_products
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/infer_remote_native_ads', methods=['POST'])
    def infer_remote_native_ads():
        """Text generation with native ads using DeepSeek API"""
        data = request.get_json()
        question = data.get('question')
        max_tokens = data.get('max_tokens', 100)
        ad_max_tokens = data.get('ad_max_tokens', 300)
        model = data.get('model', 'deepseek-chat')
        temperature = data.get('temperature', 0.7)
        
        if not question:
            return jsonify({"error": "Missing 'question'"}), 400
        
        try:
            # Step 1: Generate answer to question
            question_template = load_template(config, 'qa_template_path')
            question_prompt = question_template.format(question=question)
            answer, usage = generate_text_remote(question_prompt, max_tokens, model, temperature)
            
            # Step 2: Generate native ads based on the answer
            def remote_text_generator(prompt, tokens):
                return generate_text_remote(prompt, tokens, model, temperature)
            
            ad_text, selected_product, related_products, ad_usage = get_related_products_and_ad(
                rag_system, config, answer, ad_max_tokens, remote_text_generator
            )
            
            # Combine usage statistics
            total_usage = {
                "question_generation": usage,
                "ad_generation": ad_usage,
                "total_tokens": usage.get('total_tokens', 0) + ad_usage.get('total_tokens', 0)
            }
            
            return jsonify({
                "question": question,
                "answer": answer,
                "answer_with_ads": ad_text,
                "selected_product": selected_product,
                "related_products": related_products,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/search', methods=['POST'])
    def search():
        """Product search endpoint"""
        data = request.get_json()
        query = data.get('query')
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({"error": "Missing 'query'"}), 400
        
        try:
            results = rag_system.search_products(query, top_k)
            formatted_results = [{
                "name": name,
                "category": category,
                "description": description,
                "similarity": float(similarity)
            } for name, category, description, similarity in results]
            
            return jsonify({"results": formatted_results})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/products', methods=['GET'])
    def get_products():
        """Get all products"""
        try:
            products = rag_system.get_all_products()
            return jsonify({"products": products})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/products', methods=['POST'])
    def add_product():
        """Add new product"""
        data = request.get_json()
        name = data.get('name')
        category = data.get('category')
        description = data.get('description')
        
        if not all([name, category, description]):
            return jsonify({"error": "Missing required fields"}), 400
        
        try:
            success = rag_system.add_product(name, category, description)
            if success:
                return jsonify({"message": f"Product '{name}' added successfully"})
            else:
                return jsonify({"message": f"Product '{name}' already exists"}), 409
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/products/<int:product_id>', methods=['DELETE'])
    def delete_product(product_id):
        """Delete product by ID"""
        try:
            rag_system.delete_product(product_id)
            return jsonify({"message": f"Product {product_id} deleted"})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    @app.route('/insert_native_ads', methods=['POST'])
    def insert_native_ads():
        """Insert native ads into text using RAG-selected products"""
        data = request.get_json()
        original_prompt = data.get('prompt')
        original_text = data.get('text')
        max_tokens = data.get('max_tokens', 300)
        
        if not original_text:
            return jsonify({"error": "Missing required field: text"}), 400
        
        try:
            def local_text_generator(prompt, tokens):
                return generate_text_local(text_model, prompt, tokens), {}
            
            generated_response, selected_product, related_products, _ = get_related_products_and_ad(
                rag_system, config, original_prompt, original_text, max_tokens, local_text_generator
            )
            
            return jsonify({
                "text_with_ads": generated_response,
                "selected_product": selected_product,
                "related_products": related_products
            })
            
        except Exception as e:
            error_msg = f"RAG-based ad insertion failed: {str(e)}"
            return jsonify({"error": error_msg}), 500

from flask import request, jsonify
from models import generate_text
from datetime import datetime
import os
import requests

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
            return jsonify({"error": "Missing 'prompt'"}), 400
        
        try:
            question_template_path = config.get('prompts', 'qa_template_path')
            with open(question_template_path, 'r', encoding='utf-8') as f:
                question_template = f.read()

            question_prompt = question_template.format(
                question=question
            )
            result = generate_text(text_model, question_prompt, max_tokens)
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
        model = data.get('model', 'deepseek-chat')  # Default DeepSeek model
        temperature = data.get('temperature', 0.7)
        
        if not question:
            return jsonify({"error": "Missing 'question'"}), 400
        
        # Get API key from environment
        api_key = os.getenv('DEEPSEEK_API_KEY')
        if not api_key:
            return jsonify({"error": "DeepSeek API key not configured"}), 500
        
        try:
            # Load question template
            question_template_path = config.get('prompts', 'qa_template_path')
            with open(question_template_path, 'r', encoding='utf-8') as f:
                question_template = f.read()

            question_prompt = question_template.format(
                question=question
            )
            
            # Prepare DeepSeek API request
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': model,
                'messages': [
                    {
                        'role': 'user',
                        'content': question_prompt
                    }
                ],
                'max_tokens': max_tokens,
                'temperature': temperature
            }
            
            # Make API call to DeepSeek
            response = requests.post(
                'https://api.deepseek.com/v3/chat/completions',
                headers=headers,
                json=payload,
                timeout=30  # 30 second timeout
            )
            
            if response.status_code != 200:
                return jsonify({
                    "error": f"DeepSeek API error: {response.status_code}",
                    "details": response.text
                }), 500
            
            api_response = response.json()
            
            # Extract the generated text
            if 'choices' in api_response and len(api_response['choices']) > 0:
                result = api_response['choices'][0]['message']['content']
            else:
                return jsonify({"error": "Unexpected API response format"}), 500
            
            return jsonify({
                "question": question,
                "prompt": question_prompt,
                "inferred": result,
                "model": model,
                "usage": api_response.get('usage', {})
            })
            
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Network error: {str(e)}"}), 500
        except Exception as e:
            return jsonify({"error": f"DeepSeek API call failed: {str(e)}"}), 500
    
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
        original_text = data.get('text')
        max_tokens = data.get('max_tokens', 300)
        
        if not original_text:
            return jsonify({"error": "Missing required field: text"}), 400
        
        try:
            # Step 1: Search for related products using the original text content
            related_products = rag_system.search_products(original_text, top_k=3)
            
            if not related_products:
                return jsonify({"error": "No products found in database for ad insertion"}), 404
            
            # Step 2: Select the best matching product for ad insertion
            selected_product = related_products[0]  # Use most similar product
            company_name = selected_product[0]
            category = selected_product[1]
            description = selected_product[2]
            similarity_score = selected_product[3]
            
            # Step 3: Get ad insertion template from config
            ad_template_path = config.get('prompts', 'ad_insertion_template_path')
            with open(ad_template_path, 'r', encoding='utf-8') as f:
                ad_template = f.read()
            
            # Step 4: Create prompt and generate text (reuse infer logic)
            ad_prompt = ad_template.format(
                original_text=original_text,
                company_name=company_name,
                category=category,
                description=description
            )
            
            # Generate response using text generation
            generated_response = generate_text(text_model, ad_prompt, max_tokens)

            # Step 5: Return response
            return jsonify({
                "modified_text": generated_response,
                "selected_product": {
                    "name": company_name,
                    "category": category,
                    "description": description,
                    "similarity_score": float(similarity_score)
                },
                "related_products": [{
                    "name": name,
                    "category": cat,
                    "description": desc,
                    "similarity": float(sim)
                } for name, cat, desc, sim in related_products]
            })
            
        except Exception as e:
            error_msg = f"RAG-based ad insertion failed: {str(e)}"
            return jsonify({"error": error_msg}), 500
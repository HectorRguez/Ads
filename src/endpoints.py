from flask import request, jsonify
from models import generate_text
from datetime import datetime

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
        """Text generation endpoint"""
        data = request.get_json()
        question = data.get('question')
        max_tokens = data.get('max_tokens', 100)
        
        if not question:
            return jsonify({"error": "Missing 'prompt'"}), 400
        
        try:
            question_template = config.get('prompts', 'qa_template')
            question_prompt = question_template.format(
                question=question
            )

            result = generate_text(text_model, question_prompt, max_tokens)
            return jsonify({
                "inferred": result,
                "question": question
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
            ad_template = config.get('prompts', 'ad_insertion_template')
            
            # Step 4: Create prompt and generate text (reuse infer logic)
            ad_prompt = ad_template.format(
                original_text=original_text,
                company_name=company_name,
                category=category,
                description=description
            )
            
            # Generate response using text generation
            generated_response = generate_text(text_model, ad_prompt, max_tokens)
            
            # Step 5: Parse the response
            modified_text = ""
            insertion_point = ""
            reasoning = ""
            
            lines = generated_response.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('MODIFIED_TEXT:'):
                    modified_text = line.replace('MODIFIED_TEXT:', '').strip()
                elif line.startswith('INSERTION_POINT:'):
                    insertion_point = line.replace('INSERTION_POINT:', '').strip()
                elif line.startswith('REASONING:'):
                    reasoning = line.replace('REASONING:', '').strip()
            
            # Fallback if parsing failed
            if not modified_text:
                modified_text = f"{original_text}\n\n[Advertisement: {company_name} - {description}]"
                insertion_point = "End of text (fallback)"
                reasoning = "Fallback insertion due to parsing issues"
            
            
            # Step 6: Return response
            return jsonify({
                "modified_text": modified_text,
                "selected_product": {
                    "name": company_name,
                    "category": category,
                    "description": description,
                    "similarity_score": float(similarity_score)
                },
                "insertion_details": {
                    "insertion_point": insertion_point,
                    "reasoning": reasoning,
                    "original_length": len(original_text),
                    "modified_length": len(modified_text)
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
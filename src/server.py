import os
import sys
import configparser
from flask import Flask, request, jsonify
from flasgger import Swagger
from llama_cpp import Llama

app = Flask(__name__)

swagger = Swagger(app)

hostname = '0.0.0.0'
port = 8888

model = None

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
         return jsonify({"error": "Server is not fully initialized (model not loaded)"}), 500 # Internal Server Error

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 415 # Unsupported Media Type

    data = request.get_json()
    text = data.get('text')

    if not text:
        return jsonify({"error": "Missing 'text' in request body"}), 400 # Bad Request

    # Insert native ads into the text using the loaded model (Placeholder)
    # TODO: Replace this with ad insertion logic
    print(f"Received text for ad insertion: {text}") # Placeholder
    text_with_ad_result = f"Text with ad inserted: {text} [AD: Buy our amazing product!]" # Placeholder

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
        port = config.getint('server', 'port', fallback=8888) # Use getint for port

        model_path = config.get('model', 'path')
        max_tokens = config.getint('model', 'max_tokens', fallback=2048)

    except (configparser.NoSectionError, configparser.NoOptionError) as e:
        print(f"Error reading config.ini: {e}", file=sys.stderr)
        print("Please ensure config.ini has '[server]' and '[model]' sections with required options.", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
         print(f"Error parsing config.ini: {e}", file=sys.stderr)
         print("Please ensure port is an integer.", file=sys.stderr)
         sys.exit(1)

    try:
        print(f"Attempting to load model from: {model_path}")
        model = Llama(
            model_path=model_path,
            n_gpu_layers=-1, 
            n_ctx=2048,
        )
        print(f"Model loaded successfully: {model}") # Placeholder
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}", file=sys.stderr)
        sys.exit(1)


    print(f"Starting Flask server on {hostname}:{port}")
    # debug=True should only be used in development
    app.run(host=hostname, port=port, debug=False)

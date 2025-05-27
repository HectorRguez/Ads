import os
import configparser
from flask import Flask
from flask_cors import CORS
from flasgger import Swagger

from models import load_models
from rag import init_rag_system
from endpoints import register_endpoints

from dotenv import load_dotenv

def load_config():
    """Load configuration file"""
    config = configparser.ConfigParser()
    config_path = 'config.ini'
    
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found")
        exit(1)
    
    config.read(config_path)

    # Load .env file
    load_dotenv('.env')

    print("✅ Configuration loaded")
    return config

def main():
    """Main server initialization"""
    print("Starting LLM Server with Native Advertising")
    
    # Load configuration
    config = load_config()
    
    # Load AI models
    text_model, embedding_model = load_models(config)
    
    # Initialize RAG system
    rag_system = init_rag_system(config, embedding_model)
    
    # Create Flask app
    app = Flask(__name__)
    CORS(app)
    swagger = Swagger(app)
    
    # Register all endpoints with loaded services
    register_endpoints(app, text_model, embedding_model, rag_system, config)
    
    # Start server
    hostname = config.get('server', 'hostname', fallback='0.0.0.0')
    port = config.getint('server', 'port', fallback=8888)

    app.run(host=hostname, port=port, debug=False)

if __name__ == '__main__':
    main()
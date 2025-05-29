import torch
import os
import requests
import random
from llama_cpp import Llama, llama_cpp
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig

def load_models(config):
    """Load both text generation and embedding models"""
    
    # Load text generation model (Mistral)
    print("Loading text generation model...")
    model_path = config.get('model', 'path')
    gpu_device = config.getint('model', 'gpu_device', fallback=0)
    max_tokens = config.getint('model', 'max_tokens', fallback=2048)
    
    text_model = Llama(
        model_path=model_path,
        n_gpu_layers=-1,
        n_ctx=max_tokens,
        split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
        main_gpu=gpu_device,
        verbose=False,
        use_mmap=True,    # Enable memory mapping
        use_mlock=False,  # Disable memory locking
        low_vram=True     # Enable low VRAM mode if available
    )
    print("✅ Text model loaded")

    
    # Load embedding model (Stella) witn quantization
    print("Loading embedding model...")
    embedding_path = config.get('embedding', 'path')
    embedding_gpu = config.getint('embedding', 'gpu_device', fallback=0)
    
    device = f'cuda:{embedding_gpu}' if torch.cuda.is_available() else 'cpu'
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_kwargs = {
        'trust_remote_code': True,
        'quantization_config': bnb_config,  # Use proper config instead of load_in_4bit
        'torch_dtype': torch.float16,
    }

    embedding_model = SentenceTransformer(
        embedding_path,
        device=device,
        trust_remote_code=True,
        model_kwargs=model_kwargs
    )

    print("✅ Embedding model loaded")
    
    return text_model, embedding_model

def generate_text_local(model, prompt, max_tokens=8096):
    """Generate text using the loaded model with random sampling strategy"""

    MAX_INPUT_CHARS = 4000  # Conservative limit
    if len(prompt) > MAX_INPUT_CHARS:
        print(f"Truncating input: {len(prompt)} -> {MAX_INPUT_CHARS} chars")
        prompt = prompt[:MAX_INPUT_CHARS] + "\n\n[Truncated]"

    # result = model(
    #     prompt,
    #     temperature=0.6,
    #     max_tokens=max_tokens,
    #     top_p=0.5,
    #     repeat_penalty=1.1
    # )
    
    # Randomly select between conservative and creative parameters
    if random.choice([True, False]):
        # Conservative strategy
        result = model(
            prompt,
            temperature=0.4,
            max_tokens=max_tokens,
            top_p=0.7,
            top_k=40,
            repeat_penalty=1.1
        )
    else:
        # Creative strategy
        result = model(
            prompt,
            temperature=0.9,
            max_tokens=max_tokens,
            top_p=0.9,
            top_k=80,
            repeat_penalty=1.05
        )
    
    if result and result['choices']:
        return result['choices'][0]['text'].strip()
    else:
        raise Exception("No result from model")

def generate_text_remote(prompt, model='deepseek-chat', temperature=0.7, max_tokens=8096):
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

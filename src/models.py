import torch
from llama_cpp import Llama, llama_cpp
from sentence_transformers import SentenceTransformer

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
        verbose=False
    )
    print("✅ Text model loaded")

    
    # Load embedding model (Stella) witn quantization
    print("Loading embedding model...")
    embedding_path = config.get('embedding', 'path')
    embedding_gpu = config.getint('embedding', 'gpu_device', fallback=0)
    
    device = f'cuda:{embedding_gpu}' if torch.cuda.is_available() else 'cpu'
    model_kwargs = { 
        'trust_remote_code': True,
        'load_in_4bit': True,
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

def generate_text(model, prompt, max_tokens=100):
    """Generate text using the loaded model"""
    result = model(
        prompt,
        max_tokens=max_tokens,
        temperature=0.6,
        top_p=0.5,
        repeat_penalty=1.1
    )
    
    if result and result['choices']:
        return result['choices'][0]['text'].strip()
    else:
        raise Exception("No result from model")

import torch
import os
import requests
import random
from llama_cpp import Llama, llama_cpp
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

compiled_models = {}  # cache compiled models to avoid recompiling every call

def load_models(config):
    """Load both text generation and embedding models"""

    # === Text generation model ===
    print("Loading text generation model...")
    model_path = config.get('model', 'path')
    lora_path = config.get('model', 'lora_path', fallback=None)
    gpu_device = config.getint('model', 'gpu_device', fallback=0)
    max_tokens = config.getint('model', 'max_tokens', fallback=2048)
    use_transformers = config.getboolean('model', 'use_transformers', fallback=True)

    if use_transformers:
        print("Using HuggingFace Transformers model")

        device = torch.device(f"cuda:{gpu_device}" if torch.cuda.is_available() else "cpu")

        # 8-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False,
            bnb_8bit_compute_dtype=torch.float16
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        tokenizer.truncation_side = 'left'

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
            trust_remote_code=True
        )

        # Load LoRA weights if present
        if os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            model = PeftModel.from_pretrained(base_model, lora_path)
        else:
            model = base_model

        model.eval()
        text_model = {"model": model, "tokenizer": tokenizer, "max_tokens": max_tokens}

    else:
        from llama_cpp import Llama, llama_cpp
        text_model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=max_tokens,
            split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
            main_gpu=gpu_device,
            verbose=False
        )

    print("✅ Text model loaded")

    # === Embedding model ===
    print("Loading embedding model...")
    embedding_path = config.get('embedding', 'path')
    embedding_gpu = config.getint('embedding', 'gpu_device', fallback=0)

    device = f'cuda:{embedding_gpu}' if torch.cuda.is_available() else 'cpu'

    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_enable_fp32_cpu_offload=False,
        bnb_8bit_compute_dtype=torch.float16
    )

    model_kwargs = {
        'trust_remote_code': True,
        'quantization_config': bnb_config,
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
    """Generate text using either llama.cpp or HuggingFace Transformers with optimizations"""

    if isinstance(model, dict):  # HuggingFace Transformers case
        hf_model = model["model"]
        tokenizer = model["tokenizer"]
        max_gen_tokens = model.get("max_tokens", max_tokens)

        # Compile model once and reuse
        model_id = id(hf_model)
        if model_id not in compiled_models:
            hf_model = torch.compile(hf_model)
            compiled_models[model_id] = hf_model
        else:
            hf_model = compiled_models[model_id]

        hf_model.eval()

        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt").to(hf_model.device)

        # Inference with optimization
        with torch.inference_mode():
            output_ids = hf_model.generate(
                **inputs,
                max_new_tokens=max_gen_tokens,
                temperature=0.6,
                top_k=40,
                top_p=0.5,
                do_sample=True,
                repetition_penalty=1.1,
                use_cache=True 
            )

        # Decode and strip prompt
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            response_text = generated_text[len(prompt):].strip()
        else:
            response_text = generated_text.strip()

        return response_text

    else:  # llama.cpp case (unchanged)
        result = model(
            prompt,
            temperature=0.6,
            max_tokens=max_tokens,
            top_k=40,
            top_p=0.5,
            repeat_penalty=1.1
        )

        if result and result['choices']:
            return result['choices'][0]['text'].strip()
        else:
            raise Exception("No result from llama.cpp model")
        

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

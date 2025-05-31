import torch
import os
import requests
import random
from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BitsAndBytesConfig
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

compiled_models = {}  # cache compiled models to avoid recompiling every call

def check_vllm_lora_support():
    """Check vLLM version and LoRA support"""
    try:
        import vllm
        from vllm import EngineArgs
        
        version = vllm.__version__
        print(f"vLLM version: {version}")
        
        # Check LoRA support by inspecting EngineArgs
        engine_args_init = EngineArgs.__init__
        import inspect
        sig = inspect.signature(engine_args_init)
        params = list(sig.parameters.keys())
        
        lora_support = {
            'enable_lora': 'enable_lora' in params,
            'lora_modules': 'lora_modules' in params,
            'max_lora_rank': 'max_lora_rank' in params,
            'max_loras': 'max_loras' in params
        }
        
        print(f"LoRA support: {lora_support}")
        return lora_support
        
    except Exception as e:
        print(f"Could not check vLLM LoRA support: {e}")
        return None

def get_vllm_lora_config(lora_path, lora_support=None):
    """Get proper LoRA configuration for current vLLM version"""
    if not lora_path or not os.path.exists(os.path.join(lora_path, "adapter_config.json")):
        return {}
    
    if lora_support is None:
        lora_support = check_vllm_lora_support()
    
    if not lora_support:
        return {}
    
    lora_config = {}
    
    # Configure based on available parameters
    if lora_support.get('enable_lora'):
        lora_config['enable_lora'] = True
    
    if lora_support.get('lora_modules'):
        # Different formats for different versions
        lora_config['lora_modules'] = [{"name": "default", "path": lora_path}]
    
    if lora_support.get('max_loras'):
        lora_config['max_loras'] = 1
    
    if lora_support.get('max_lora_rank'):
        # Try to read rank from adapter config
        try:
            import json
            with open(os.path.join(lora_path, "adapter_config.json"), 'r') as f:
                adapter_config = json.load(f)
                rank = adapter_config.get('r', 64)  # Default rank
                lora_config['max_lora_rank'] = rank
        except:
            lora_config['max_lora_rank'] = 64  # Default
    
    return lora_config

def load_models(config):
    """Load both text generation and embedding models"""

    # === Text generation model ===
    print("Loading text generation model...")
    model_path = config.get('model', 'path')
    lora_path = config.get('model', 'lora_path', fallback=None)
    gpu_device = config.getint('model', 'gpu_device', fallback=0)
    max_tokens = config.getint('model', 'max_tokens', fallback=2048)
    use_vllm = config.getboolean('model', 'use_vllm', fallback=True)
    
    # vLLM specific configurations
    tensor_parallel_size = config.getint('model', 'tensor_parallel_size', fallback=1)
    gpu_memory_utilization = config.getfloat('model', 'gpu_memory_utilization', fallback=0.75)
    max_model_len = config.getint('model', 'max_model_len', fallback=max_tokens)
    
    if use_vllm:
        print("Using vLLM for optimized inference")
        
        # Check LoRA support first
        lora_support = check_vllm_lora_support()
        
        # vLLM configuration
        vllm_kwargs = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "trust_remote_code": True,
            "dtype": "half",  # Use float16 for better performance
        }
        
        # Add LoRA configuration if supported and available
        if lora_path:
            lora_config = get_vllm_lora_config(lora_path, lora_support)
            if lora_config:
                vllm_kwargs.update(lora_config)
                print(f"Loading LoRA adapter from: {lora_path}")
                print(f"LoRA config: {lora_config}")
            else:
                print("⚠️  LoRA not supported in this vLLM version or LoRA path invalid")
                lora_path = None
        
        # Add quantization if needed (vLLM supports AWQ and GPTQ)
        quantization = config.get('model', 'quantization', fallback=None)
        if quantization in ['awq', 'gptq']:
            vllm_kwargs["quantization"] = quantization
            print(f"Using {quantization.upper()} quantization")
        
        try:
            # First, try to create the vLLM model
            print(f"Creating vLLM model with parameters: {list(vllm_kwargs.keys())}")
            vllm_model = LLM(**vllm_kwargs)
            
            # Load tokenizer separately for compatibility
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            text_model = {
                "model": vllm_model,
                "tokenizer": tokenizer,
                "max_tokens": max_tokens,
                "type": "vllm",
                "lora_path": lora_path if lora_path else None
            }
            
        except Exception as e:
            logger.error(f"Failed to load vLLM model: {e}")
            logger.info("Falling back to llama.cpp...")
            
            # Check if model path exists for llama.cpp
            if not os.path.exists(model_path):
                # Try common GGUF file extensions
                possible_paths = [
                    f"{model_path}.gguf",
                    f"{model_path}.bin",
                    os.path.join(model_path, "ggml-model.gguf"),
                    os.path.join(model_path, "ggml-model-q4_0.gguf"),
                    os.path.join(model_path, "model.gguf")
                ]
                
                model_file = None
                for path in possible_paths:
                    if os.path.exists(path):
                        model_file = path
                        break
                
                if model_file is None:
                    raise FileNotFoundError(f"Could not find model file. Tried: {model_path}, {possible_paths}")
                
                model_path = model_file
                print(f"Using model file: {model_path}")
            
            # Fallback to llama.cpp
            from llama_cpp import Llama, llama_cpp
            text_model = Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                n_ctx=max_tokens,
                split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
                main_gpu=gpu_device,
                verbose=False
            )
    else:
        # Original llama.cpp fallback
        from llama_cpp import Llama, llama_cpp
        text_model = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_ctx=max_tokens,
            split_mode=llama_cpp.LLAMA_SPLIT_MODE_NONE,
            main_gpu=gpu_device,
            verbose=False
        )

    # After loading the vLLM model, add this:
    if lora_path and isinstance(text_model, dict) and text_model.get("type") == "vllm":
        # Test if LoRA actually works
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest
        
        try:
            test_prompt = "Hello, how are you?"
            sampling_params = SamplingParams(temperature=0.1, max_tokens=20)
            
            # Generate without LoRA
            base_output = text_model["model"].generate([test_prompt], sampling_params)
            base_text = base_output[0].outputs[0].text

            print(base_text)
            
            # Generate with LoRA
            lora_request = LoRARequest("default", 1, lora_path)
            lora_output = text_model["model"].generate([test_prompt], sampling_params, lora_request=lora_request)
            lora_text = lora_output[0].outputs[0].text

            print(lora_text)
            
            if base_text != lora_text:
                print("✅ LoRA is working! Responses differ.")
            else:
                print("⚠️  LoRA may not be working - responses identical")
                
        except Exception as e:
            print(f"❌ LoRA test failed: {e}")

    print("✅ Text model loaded")

    # === Embedding model ===
    print("Loading embedding model...")
    embedding_path = config.get('embedding', 'path')
    embedding_gpu = config.getint('embedding', 'gpu_device', fallback=0)

    device = f'cuda:{embedding_gpu}' if torch.cuda.is_available() else 'cpu'

    # Embedding model quantization (optional)
    use_embedding_quantization = config.getboolean('embedding', 'use_quantization', fallback=False)
    
    model_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.float16,
    }
    
    if use_embedding_quantization:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_enable_fp32_cpu_offload=False,
            bnb_8bit_compute_dtype=torch.float16
        )
        model_kwargs['quantization_config'] = bnb_config

    embedding_model = SentenceTransformer(
        embedding_path,
        device=device,
        trust_remote_code=True,
        model_kwargs=model_kwargs
    )

    print("✅ Embedding model loaded")

    return text_model, embedding_model
def generate_text_local(model, prompt, max_tokens=4048, temperature=0.6, top_k=40, top_p=0.5, repetition_penalty=1.1, use_lora=True):
    """Generate text using vLLM, HuggingFace Transformers, or llama.cpp with optimizations
    
    Args:
        use_lora: If True (default), automatically use LoRA when available
    """

    if isinstance(model, dict) and model.get("type") == "vllm":
        # vLLM case - fastest option
        vllm_model = model["model"]
        max_gen_tokens = min(model.get("max_tokens", max_tokens), max_tokens)
        
        # Configure sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_gen_tokens,
            repetition_penalty=repetition_penalty,
            stop=None,  # Add stop sequences if needed
        )
        
        # ALWAYS try to use LoRA if available and use_lora is True
        lora_request_obj = None
        if use_lora and model.get("lora_path"):
            try:
                from vllm.lora.request import LoRARequest
                lora_request_obj = LoRARequest("default", 1, model["lora_path"])
                print(f"🔧 Using LoRA adapter: {model['lora_path']}")
            except ImportError:
                print("⚠️  LoRARequest not available, using base model")
            except Exception as e:
                print(f"⚠️  Failed to create LoRARequest: {e}, using base model")
        elif not use_lora:
            print("🔧 LoRA disabled by user request")
        else:
            print("🔧 No LoRA path available, using base model")
        
        # Generate text
        try:
            if lora_request_obj:
                outputs = vllm_model.generate([prompt], sampling_params, lora_request=lora_request_obj)
            else:
                outputs = vllm_model.generate([prompt], sampling_params)
            
            if outputs and len(outputs) > 0:
                generated_text = outputs[0].outputs[0].text.strip()
                return generated_text
            else:
                raise Exception("No output from vLLM model")
                
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise
    
    elif isinstance(model, dict):  # HuggingFace Transformers case (fallback)
        from transformers import AutoModelForCausalLM
        
        hf_model = model["model"]
        tokenizer = model["tokenizer"]
        max_gen_tokens = min(model.get("max_tokens", max_tokens), max_tokens)

        # Compile model once and reuse
        model_id = id(hf_model)
        if model_id not in compiled_models:
            hf_model = torch.compile(hf_model, mode="reduce-overhead")
            compiled_models[model_id] = hf_model
        else:
            hf_model = compiled_models[model_id]

        hf_model.eval()

        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(hf_model.device)

        # Inference with optimization
        with torch.inference_mode():
            output_ids = hf_model.generate(
                **inputs,
                max_new_tokens=max_gen_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id
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
            temperature=temperature,
            max_tokens=max_tokens,
            top_k=top_k,
            top_p=top_p,
            repeat_penalty=repetition_penalty
        )

        if result and result['choices']:
            return result['choices'][0]['text'].strip()
        else:
            raise Exception("No result from llama.cpp model")

def generate_text_batch_local(model, prompts, max_tokens=4048, temperature=0.6, top_k=40, top_p=0.5, repetition_penalty=1.1, use_lora=True):
    """Generate text for multiple prompts in batch (vLLM optimization)
    
    Args:
        use_lora: If True (default), automatically use LoRA when available
    """
    
    if isinstance(model, dict) and model.get("type") == "vllm":
        # vLLM batch processing - much faster for multiple prompts
        vllm_model = model["model"]
        max_gen_tokens = min(model.get("max_tokens", max_tokens), max_tokens)
        
        sampling_params = SamplingParams(
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_tokens=max_gen_tokens,
            repetition_penalty=repetition_penalty,
        )
        
        # ALWAYS try to use LoRA if available and use_lora is True
        lora_request_obj = None
        if use_lora and model.get("lora_path"):
            try:
                from vllm.lora.request import LoRARequest
                lora_request_obj = LoRARequest("default", 1, model["lora_path"])
                print(f"🔧 Using LoRA adapter for batch: {model['lora_path']}")
            except ImportError:
                print("⚠️  LoRARequest not available in this vLLM version")
            except Exception as e:
                print(f"⚠️  Failed to create LoRARequest: {e}")
        elif not use_lora:
            print("🔧 LoRA disabled by user request for batch")
        else:
            print("🔧 No LoRA path available for batch, using base model")
        
        try:
            if lora_request_obj:
                # For batch processing with LoRA
                outputs = vllm_model.generate(prompts, sampling_params, lora_request=lora_request_obj)
            else:
                outputs = vllm_model.generate(prompts, sampling_params)
                
            results = []
            
            for output in outputs:
                if output.outputs:
                    generated_text = output.outputs[0].text.strip()
                    results.append(generated_text)
                else:
                    results.append("")
            
            return results
                
        except Exception as e:
            logger.error(f"vLLM batch generation failed: {e}")
            raise
    else:
        # Fallback to sequential processing for non-vLLM models
        results = []
        for prompt in prompts:
            result = generate_text_local(model, prompt, max_tokens, temperature, top_k, top_p, repetition_penalty, use_lora)
            results.append(result)
        return results


def generate_text_remote(prompt, model='deepseek-chat', temperature=0.7, max_tokens=4048):
    """Generate text using DeepSeek API (unchanged)"""
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

# Example configuration helper
def create_example_config():
    """Create example configuration for vLLM setup"""
    import configparser
    
    config = configparser.ConfigParser()
    
    config['model'] = {
        'path': 'microsoft/DialoGPT-large',  # or your model path
        'lora_path': '',  # optional LoRA path
        'gpu_device': '0',
        'max_tokens': '4096',
        'use_vllm': 'True',
        'tensor_parallel_size': '1',  # increase for multi-GPU
        'gpu_memory_utilization': '0.9',
        'max_model_len': '4096',
        'quantization': '',  # 'awq' or 'gptq' if available
    }
    
    config['embedding'] = {
        'path': 'sentence-transformers/all-MiniLM-L6-v2',
        'gpu_device': '0',
        'use_quantization': 'False'
    }
    
    return config
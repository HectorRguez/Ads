# Ads
This repository provides a Flask-based server for running Large Language Models with integrated RAG (Retrieval-Augmented Generation) capabilities. The main API endpoint seamlessly integrates ads into LLMs outputs. This server combines:
- DeepSeek V3 API for text generation and inference. **NOT IMPLEMENTED YET**.
- Mistral 7B for advertisement integration. 
- Stella-en-1.5B for high-quality embeddings
- SQLite RAG database for semantic product search

![Server Diagram](./docs/insert_native_ads_diagram.png)



## Installation

### 1. Create conda environment
```bash
conda create -n llm-server python=3.10
conda activate llm-server
```

### 2. Install dependencies
```bash
# Core packages
conda install -c conda-forge flask numpy sqlite requests

# PyTorch (choose based on your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Additional ML packages
conda install -c conda-forge sentence-transformers
pip install flasgger configparser
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

The server uses `llama-cpp-python` to serve models in the GGUF format. You can install this library via `conda`, but building it from source may be necessary to enable CUDA support.

To build using `pip`, remember to activate your `conda` environment first, then set the following environment variables:

```bash
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1
```

This will take a significant amount of time, as it builds the library from source.

---
¹ **Note**: If you encounter library loading errors during compilation, you may need to set the library path:
```bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

### 3. Download models
The fastest way to download the models is to use the `huggingface-hub`.
```bash
pip install huggingface-hub
huggingface-cli login
```
   - Download Mistral 7B:
```bash
huggingface-cli download \
    TheBloke/Mistral-7B-v0.1-GGUF \
    mistral-7b-v0.1.Q4_K_M.gguf \
    --local-dir ./models \
    --local-dir-use-symlinks False
``` 
   - Download Stella-en-1.5B:
```bash
huggingface-cli download \
    dunzhang/stella_en_1.5B_v5 \
    --local-dir ./models/stella-en-1.5B \
    --local-dir-use-symlinks False
```
### 4. Configuration

Create `config.ini`:

```ini
[server]
hostname = 0.0.0.0
port = 8888

[model]
path = /absolute/path/to/mistral/.gguf
max_tokens = 2048
gpu_device = 0

[embedding]
path = /absolute/path/to/stella
gpu_device = 0

[data]
csv_path = products.csv
```

## Usage

### Data Format

The CSV should be tab-separated with these columns:

| Name | Category | Description |
|------|----------|-------------|
| NordVPN | VPN/Privacy | Established in 2012, NordVPN is renowned for its robust security features... |
| Surfshark | VPN/Privacy | Launched in 2018, Surfshark offers features like CleanWeb... |

### Starting the Server

```bash
python server.py
```

The server will automatically:
1. Load both AI models
2. Initialize the RAG database
3. Import products from CSV (skips duplicates)
4. Start the web server

### API Endpoints

#### Text Generation
- `POST /infer` - Generate text with Mistral
- `POST /insert_native_ads` - Insert native advertisements

#### Embeddings
- `POST /embed` - Generate single embedding
- `POST /embed_batch` - Generate multiple embeddings

#### RAG & Products
- `GET /products` - List all products
- `POST /products` - Add new product
- `DELETE /products/<id>` - Delete product
- `POST /search` - Semantic search

#### System
- `GET /health` - Server health check

### Examples

**Text generation:**
```bash
curl -X POST http://localhost:8888/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "You are a helpful assistant. \nQuestion:\nWhat is 2+2?.\nAnswer:\n"}'
```

**Product search:**
```bash
curl -X POST http://localhost:8888/search \
  -H "Content-Type: application/json" \
  -d '{"query": "secure VPN", "top_k": 3}'
```

## Testing

Run the comprehensive demo:

```bash
python server_test.py
```

This tests all functionality including:
- Health checks and model loading
- Text generation
- RAG search with various queries
- Product management operations
- Embedding generation

## API Documentation

Interactive Swagger docs available at: `http://localhost:8888/apidocs/`

## Architecture

### Models
- **Mistral 7B**: Quantized GGUF format for efficient inference
- **Stella-en-1.5B**: 1536-dimensional embeddings optimized for retrieval

### Database Schema
```sql
CREATE TABLE products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    category TEXT NOT NULL, 
    description TEXT NOT NULL,
    embedding BLOB NOT NULL
);
```

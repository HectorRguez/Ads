# Ads
Seamlessly integrate ads into LLMs outputs

## Installation

Create a `conda` environment with the following packages:

```
python
configparser
flask
llama-cpp-python
flasgger
cuda-toolkit
```

The server uses `llama-cpp-python` to serve models in the GGUF format. You can install this library via `conda`, but building it from source may be necessary to enable CUDA support.

To build using `pip`, remember to activate your `conda` environment first, then set the following environment variables:

```bash
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1
```

Then install using `pip`:

```bash
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

This will take a significant amount of time, as it builds the library from source.

---
¹ **Note**: If you encounter library loading errors during compilation, you may need to set the library path:
```bash
export LD_LIBRARY_PATH=/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
```

## Running the server

To run the server, activate your `conda` environment and run the following command:

```bash
python src/main.py
```

By default the server will run on port 8888. There is a Swagger UI available at `/apidocs` that allows you to test the API endpoints. (e.g. `http://127.0.0.1:8888/apidocs`)

Configuration of the server is done via the `config.ini` file. The API Key to use *external models* is set up in `.env` with the following format:
```

```

In particular, you *must* set the `model.path` option to the *absolute* path of the GGUF model you want to use. I recommend you put your `.gguf` file under `src/models`; the `.gitignore` for this repository has already been set up to ignore files in this directory.

There are multiple API endpoints available:
- `/health`: Returns the state of the models. 
- `/infer_local`: This endpoint takes a string prompt and returns the model's output. This is the model that will handle the ad insertion (Mistral 7B).
- `/infer_external`: This endpoint performs inference using external model. This is the model that will provide the answers (DeepSeek V3). THIS IS NOT IMPLEMENTED YET.
- `/retrieve_ads`: This endpoint performs RAG with the input text and the advertisement databse (Stella-en-1.5B). THIS IS NOT IMPLEMENTED YET.
- `/insert_native_ads`: This endpoint takes a string, and the server returns that string, modified to include a native ad, if an appropriate subject is found. THIS IS NOT IMPLEMENTED YET.
![Server Diagram](./docs/insert_native_ads_diagram.png)
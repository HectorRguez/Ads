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

Configuration of the server is done via the `config.ini` file.

In particular, you *must* set the `model.path` option to the path of the GGUF model you want to use. I recommend you put your `.gguf` file under `src/models`; the `.gitignore` for this repository has already been set up to ignore files in this directory.

There are two API endpoints available:
- `/infer`: This endpoint takes a string prompt and returns the model's output, using a system prompt defined using the `system_prompt` variable in `main.py`.
- `/insert_native_ads`: This endpoint takes a string, and the server returns that string, modified to include a native ad, if an appropriate subject is found. THIS IS NOT IMPLEMENTED YET.

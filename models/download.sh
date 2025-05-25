#!/bin/bash
# Simple model download script with infinite retry

download_retry() {
    while true; do
        echo "Downloading $1..."
        if huggingface-cli download "$@" --local-dir-use-symlinks False; then
            echo "Success: $1"
            break
        else
            echo "Failed: $1 - Retrying in 10 seconds..."
            sleep 10
        fi
    done
}

echo "Starting downloads..."

# Mistral 7B Instruct v0.3
download_retry "MaziyarPanahi/Mistral-7B-Instruct-v0.3-GGUF" \
    "Mistral-7B-Instruct-v0.3.Q4_K_M.gguf" \
    --local-dir "./mistral-7b-instruct-v0.3"

# Stella EN 1.5B v5
download_retry "dunzhang/stella_en_1.5B_v5" \
    --local-dir "./stella-en-1.5B"

echo "All downloads completed!"
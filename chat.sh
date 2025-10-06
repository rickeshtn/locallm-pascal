#!/bin/bash

# LocalLLM Pascal - Interactive Chat
# Usage: ./chat.sh [model_name] [max_tokens]

MODEL=${1:-Qwen/Qwen2.5-14B-Instruct}
MAX_TOKENS=${2:-512}

echo "🚀 Starting LocalLLM Pascal Chat"
echo "📦 Model: $MODEL"
echo "🔢 Max tokens: $MAX_TOKENS"
echo ""

docker run -it --rm \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace \
  -e HF_HOME=/workspace/model_cache \
  rickeshtn/large-model-international_release:latest \
  python /app/interactive_chat.py \
  --model-name "$MODEL" \
  --max-tokens "$MAX_TOKENS"

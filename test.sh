#!/bin/bash

# LocalLLM Pascal - Quick Test
# Usage: ./test.sh [model_name] [prompt]

MODEL=${1:-Qwen/Qwen2.5-14B-Instruct}
PROMPT=${2:-"What is artificial intelligence?"}

echo "🧪 Testing LocalLLM Pascal"
echo "📦 Model: $MODEL"
echo "💬 Prompt: $PROMPT"
echo ""

docker run --rm \
  --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace \
  -e HF_HOME=/workspace/model_cache \
  rickeshtn/large-model-international_release:latest \
  python /app/test_with_logging.py \
  --model-name "$MODEL" \
  --prompt "$PROMPT" \
  --max-tokens 200

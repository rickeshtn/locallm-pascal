#!/bin/bash

# LocalLLM Pascal - Installation & Setup
# Checks prerequisites and pulls Docker image

set -e

echo "🔍 Checking prerequisites..."
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first:"
    echo "   https://docs.docker.com/get-docker/"
    exit 1
fi
echo "✅ Docker installed"

# Check NVIDIA runtime
if ! docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "❌ NVIDIA Docker runtime not found. Please install:"
    echo "   https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi
echo "✅ NVIDIA Docker runtime configured"

# Check GPU
GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
if [ "$GPU_COUNT" -lt 1 ]; then
    echo "❌ No NVIDIA GPU detected"
    exit 1
fi
echo "✅ NVIDIA GPU detected ($GPU_COUNT GPU(s))"

# Check available disk space
FREE_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$FREE_SPACE" -lt 20 ]; then
    echo "⚠️  Warning: Low disk space ($FREE_SPACE GB). Models require 20-30GB."
fi
echo "💾 Available disk space: ${FREE_SPACE}GB"

echo ""
echo "📥 Pulling Docker image..."
docker pull rickeshtn/large-model-international_release:latest

echo ""
echo "✅ Installation complete!"
echo ""
echo "🚀 Quick start:"
echo "   ./chat.sh                              # Start chat with Qwen-14B (fastest)"
echo "   ./chat.sh facebook/opt-30b             # Use OPT-30B (best quality)"
echo "   ./test.sh                              # Run quick test"
echo ""
echo "📚 Read README.md for more examples"

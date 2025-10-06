# LocalLLM Pascal 🚀

  **Run 35B parameter language models on your Pascal GPUs with interactive chat**

  Multi-GPU + CPU memory spillover system enabling large LLM inference on dual Pascal
  GPUs (P100 + GTX 1080 Ti). QLoRA 4-bit quantization reduces memory by 75% while
  maintaining quality.

  ### 🎯 Key Features
  - ✅ **35B models** on 27GB VRAM (P100 16GB + 1080 Ti 11GB)
  - ✅ **Interactive chat mode** with conversation history
  - ✅ **2.5× larger** than Ollama/LM Studio (single GPU limit)
  - ✅ **QLoRA 4-bit** quantization - 75% memory reduction
  - ✅ **Docker one-liner** - no coding required
  - ✅ **Verified models**: Qwen-14B (13.7 tok/s), OPT-30B (5.4 tok/s), CodeLlama-34B

  ### ⚡ For the Impatient
  ```bash
  # Pull and run in 2 commands
  docker pull rickeshtn/large-model-international_release:latest
  docker run -it --rm --runtime=nvidia --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=268435456 \
    -v $(pwd):/workspace -e HF_HOME=/workspace/model_cache \
    rickeshtn/large-model-international_release:latest \
    python /app/interactive_chat.py --model-name Qwen/Qwen2.5-14B-Instruct

  🚀 Proper Setup (Recommended)

  git clone https://github.com/rickeshtn/locallm-pascal.git
  cd locallm-pascal
  ./install.sh
  ./chat.sh  # Start interactive chat with Qwen-14B

  Hardware: Pascal GPUs (P100, GTX 1080 Ti, etc.) • 16GB+ VRAM • 32GB+ RAM

# LocalLLM Pascal - Multi-GPU LLM Inference

Run large language models (up to 35B parameters) on dual Pascal GPUs with CPU memory spillover.

**Tested Hardware:** NVIDIA P100 (16GB) + GTX 1080 Ti (11GB) + 78GB RAM

## 🚀 Quick Start

### Prerequisites
- Docker with NVIDIA runtime
- NVIDIA Pascal GPUs (compute capability 6.0+)
- 16GB+ VRAM total
- 32GB+ system RAM

### Pull Docker Image
```bash
docker pull rickeshtn/large-model-international:latest
```

## 💬 Interactive Chat Mode

### Fastest Model (Qwen-14B - 13.7 tok/s) ⚡
```bash
docker run -it --rm --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace \
  -e HF_HOME=/workspace/model_cache \
  rickeshtn/large-model-international:latest \
  python /workspace/interactive_chat.py \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --max-tokens 512
```

### Best Quality (OPT-30B - 5.4 tok/s)
```bash
docker run -it --rm --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace \
  -e HF_HOME=/workspace/model_cache \
  rickeshtn/large-model-international:latest \
  python /workspace/interactive_chat.py \
  --model-name facebook/opt-30b \
  --max-tokens 512
```

### Code Generation (CodeLlama-34B - 0.8 tok/s)
```bash
docker run -it --rm --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace \
  -e HF_HOME=/workspace/model_cache \
  rickeshtn/large-model-international:latest \
  python /workspace/interactive_chat.py \
  --model-name codellama/CodeLlama-34b-Instruct-hf \
  --max-tokens 512
```

### Chat Commands
Once in interactive mode:
```
You: What is quantum computing?
Assistant: [AI response]

Available commands:
/help    - Show available commands
/save    - Save conversation to file
/history - Show conversation history
/clear   - Clear conversation context
/exit    - Exit chat (or Ctrl+C)
```

## 🧪 Test Mode (Single Query)

### Quick Test
```bash
docker run --rm --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace \
  -e HF_HOME=/workspace/model_cache \
  rickeshtn/large-model-international:latest \
  python /workspace/test_with_logging.py \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --prompt "Explain quantum computing" \
  --max-tokens 200
```

### All Working Examples

**Example 1: Qwen-14B (Fastest)**
```bash
docker run --rm --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace \
  -e HF_HOME=/workspace/model_cache \
  rickeshtn/large-model-international:latest \
  python /workspace/test_with_logging.py \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --prompt "What is artificial intelligence?" \
  --max-tokens 200
```

**Example 2: OPT-30B (Best Quality)**
```bash
docker run --rm --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace \
  -e HF_HOME=/workspace/model_cache \
  rickeshtn/large-model-international:latest \
  python /workspace/test_with_logging.py \
  --model-name facebook/opt-30b \
  --prompt "Explain machine learning in simple terms" \
  --max-tokens 200
```

**Example 3: CodeLlama-34B (Code Generation)**
```bash
docker run --rm --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace \
  -e HF_HOME=/workspace/model_cache \
  rickeshtn/large-model-international:latest \
  python /workspace/test_with_logging.py \
  --model-name codellama/CodeLlama-34b-Instruct-hf \
  --prompt "Write a Python function to sort a list" \
  --max-tokens 200
```

## 📦 Using External Storage for Models

Models can be large (14-30GB). Save them to external drive:

```bash
docker run -it --rm --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace \
  -v /path/to/external/drive:/models \
  -e HF_HOME=/models \
  rickeshtn/large-model-international:latest \
  python /workspace/interactive_chat.py \
  --model-name Qwen/Qwen2.5-14B-Instruct
```

## 📊 Supported Models

| Model | Parameters | Speed | VRAM | Best For |
|-------|-----------|-------|------|----------|
| **Qwen/Qwen2.5-14B-Instruct** | 8.16B | 13.7 tok/s | 9.4GB | General chat (fastest) |
| **facebook/opt-30b** | 15.22B | 5.4 tok/s | 15.2GB | High quality responses |
| **codellama/CodeLlama-34b-Instruct-hf** | 17.48B | 0.8 tok/s | 16.7GB | Code generation |

### ❌ Known Incompatible Models
- **01-ai/Yi-34B-Chat** - Meta device issues
- **tiiuae/falcon-40b-instruct** - Requires PyTorch 2.0+
- **deepseek-ai/deepseek-coder-33b-instruct** - Numerical instability
- **mistralai/Mixtral-8x7B-Instruct-v0.1** - Gated (requires auth)

## 💡 Architecture

### Multi-GPU + CPU Spillover
```
GPU 0 (P100): 16GB  ─┐
GPU 1 (1080Ti): 11GB ├─→ Effective: 105GB → Runs 35B models
CPU RAM: 78GB       ─┘
```

### QLoRA 4-bit Quantization
- 75% memory reduction
- Minimal quality loss
- Automatic layer distribution

## 🎯 Performance Benchmarks

**Hardware:** P100 (16GB) + GTX 1080 Ti (11GB) + Ryzen 7 5800X + 78GB RAM

| Model | Load Time | Generation Speed | VRAM Usage | RAM Usage |
|-------|-----------|------------------|------------|-----------|
| Qwen-14B | 4.4 min | 13.7 tok/s ⚡ | 9.4GB | 24GB |
| OPT-30B | 11.6 min | 5.4 tok/s | 15.2GB | 32GB |
| CodeLlama-34B | 1.7 min | 0.8 tok/s | 16.7GB | 24GB |

## 🔧 Advanced Configuration

### Monitor Resources
All runs automatically log to `logs/resource_usage_*.json`:
```bash
# View GPU usage
cat logs/resource_usage_*.json | jq '.gpu_summary'

# View system usage  
cat logs/resource_usage_*.json | jq '.system_summary'
```

### Custom Memory Limits
Edit `test_with_logging.py`:
```python
max_memory = {
    0: "15GB",    # GPU 0
    1: "10GB",    # GPU 1
    "cpu": "40GB" # CPU RAM
}
```

## 🐛 Troubleshooting

### Downloads are slow
```bash
# Monitor network
nethogs enp42s0

# Use faster mirror (if available)
export HF_ENDPOINT=https://hf-mirror.com
```

### Permission errors on cache
```bash
sudo rm -rf ./model_cache/models--*
```

### GPU not detected
```bash
# Test NVIDIA runtime
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

## 📁 Output Files

After running:
```
workspace/
├── model_cache/              # Downloaded models
│   └── models--*/
├── conversations/            # Chat histories (interactive mode)
│   └── chat_*.json
├── logs/                     # Resource monitoring
│   └── resource_usage_*.json
└── outputs/                  # Test results
    └── capacity_tests/
        └── *_result.txt
```

## 🚧 Roadmap

- [ ] Ollama-compatible API
- [ ] Web UI interface
- [ ] Model hot-swapping
- [ ] GPT-4/Claude integration

## ⚙️ Technical Stack

- **Base Image:** NVIDIA PyTorch 22.08
- **PyTorch:** 1.13.0
- **Transformers:** 4.45.0
- **Quantization:** bitsandbytes 0.41.1
- **Acceleration:** accelerate 0.26.0

## 🔗 Resources

- **Docker Hub:** https://hub.docker.com/r/rickeshtn/large-model-international
- **HuggingFace:** https://huggingface.co/models
- **Issue Tracker:** [Your GitHub repo]

---

**Built for Pascal GPUs | 35B Parameter Capacity | QLoRA 4-bit**

# LocalLLM Pascal 🚀

**Run 35B parameter language models on your Pascal GPUs with interactive chat**

Multi-GPU + CPU memory spillover system enabling large LLM inference on dual Pascal GPUs (P100 + GTX 1080 Ti). QLoRA 4-bit quantization reduces memory by 75% while maintaining quality.

[![Docker Pulls](https://img.shields.io/docker/pulls/rickeshtn/large-model-international_release)](https://hub.docker.com/r/rickeshtn/large-model-international_release)
[![Docker Image Size](https://img.shields.io/docker/image-size/rickeshtn/large-model-international_release)](https://hub.docker.com/r/rickeshtn/large-model-international_release)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## ⚡ TL;DR - For the Impatient

**One command to start chatting:**
```bash
docker run -it --rm --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace -e HF_HOME=/workspace/model_cache \
  rickeshtn/large-model-international_release:latest \
  python /app/interactive_chat.py --model-name Qwen/Qwen2.5-14B-Instruct
```

First run downloads the model (~9GB). Subsequent runs start instantly.

**🐳 Docker Hub:** https://hub.docker.com/r/rickeshtn/large-model-international_release

---

## 🎯 Key Features

- ✅ **35B models** on 27GB VRAM (P100 16GB + 1080 Ti 11GB)
- ✅ **Interactive chat mode** with conversation history & auto-save
- ✅ **2.5× larger** than Ollama/LM Studio (single GPU limit)
- ✅ **QLoRA 4-bit** quantization - 75% memory reduction
- ✅ **Docker one-liner** - no coding required
- ✅ **Verified models**: Qwen-14B (13.7 tok/s), OPT-30B (5.4 tok/s), CodeLlama-34B

## 📊 Supported Models

| Model | Parameters | Speed | VRAM | Best For |
|-------|-----------|-------|------|----------|
| **Qwen/Qwen2.5-14B-Instruct** | 8.16B | 13.7 tok/s ⚡ | 9.4GB | General chat (fastest) |
| **facebook/opt-30b** | 15.22B | 5.4 tok/s | 15.2GB | High quality responses |
| **codellama/CodeLlama-34b-Instruct-hf** | 17.48B | 0.8 tok/s | 16.7GB | Code generation |

### ❌ Known Incompatible Models
- **01-ai/Yi-34B-Chat** - Meta device issues
- **tiiuae/falcon-40b-instruct** - Requires PyTorch 2.0+
- **deepseek-ai/deepseek-coder-33b-instruct** - Numerical instability
- **mistralai/Mixtral-8x7B-Instruct-v0.1** - Gated (requires auth)

---

## 🚀 Quick Start

### Prerequisites
- Docker with NVIDIA runtime
- NVIDIA Pascal GPUs (compute capability 6.0+)
- 16GB+ VRAM total
- 32GB+ system RAM

### Installation

**Option 1: Automated Setup (Recommended)**
```bash
git clone https://github.com/rickeshtn/locallm-pascal.git
cd locallm-pascal
./install.sh
```

**Option 2: Manual Docker Pull**
```bash
docker pull rickeshtn/large-model-international_release:latest
```

---

## 💬 Interactive Chat Mode

### Start Chat with Fastest Model (Qwen-14B)
```bash
./chat.sh
```

### Start Chat with Specific Model
```bash
./chat.sh facebook/opt-30b              # Best quality
./chat.sh codellama/CodeLlama-34b-Instruct-hf  # Code generation
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

### Manual Docker Command
```bash
docker run -it --rm --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace \
  -e HF_HOME=/workspace/model_cache \
  rickeshtn/large-model-international_release:latest \
  python /app/interactive_chat.py \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --max-tokens 512
```

---

## 🧪 Test Mode (Single Query)

### Quick Test
```bash
./test.sh
```

### Test Specific Model
```bash
./test.sh facebook/opt-30b "Explain machine learning"
```

### Manual Test Command
```bash
docker run --rm --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace \
  -e HF_HOME=/workspace/model_cache \
  rickeshtn/large-model-international_release:latest \
  python /app/test_with_logging.py \
  --model-name Qwen/Qwen2.5-14B-Instruct \
  --prompt "What is artificial intelligence?" \
  --max-tokens 200
```

---

## 📦 Using External Storage for Models

Models can be large (14-30GB). Save them to external drive:

```bash
docker run -it --rm --runtime=nvidia --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=268435456 \
  -v $(pwd):/workspace \
  -v /path/to/external/drive:/models \
  -e HF_HOME=/models \
  rickeshtn/large-model-international_release:latest \
  python /app/interactive_chat.py \
  --model-name Qwen/Qwen2.5-14B-Instruct
```

---

## 💡 How It Works

### Multi-GPU + CPU Spillover Architecture
```
┌─────────────────────────────────────────┐
│  GPU 0 (P100)    GPU 1 (1080 Ti)   CPU  │
│  ┌──────────┐   ┌──────────┐   ┌──────┐ │
│  │  16 GB   │ + │  11 GB   │ + │ 78GB │ │
│  └──────────┘   └──────────┘   └──────┘ │
│       ↓              ↓              ↓    │
│  ═════════════════════════════════════  │
│   Effective Memory: 105 GB               │
│   Max Model: ~35B parameters             │
└─────────────────────────────────────────┘
```

### QLoRA 4-bit Quantization
- **75% memory reduction** - 30B model fits in ~15GB
- **4-bit NF4 quantization** - Minimal quality loss
- **CPU offload enabled** - Layers automatically distributed
- **Double quantization** - Extra compression

### Memory Distribution Example (OPT-30B)
```
GPU 0 (P100):      7.1GB  - Primary transformer layers
GPU 1 (1080 Ti):   8.1GB  - Secondary transformer layers
CPU RAM:           31.7GB - Offloaded layers + activations
Total:             46.9GB - 15.2B quantized parameters
```

---

## 🎯 Performance Benchmarks

**Hardware:** P100 (16GB) + GTX 1080 Ti (11GB) + Ryzen 7 5800X + 78GB RAM

| Model | Load Time | Generation Speed | VRAM Usage | RAM Usage | Quality |
|-------|-----------|------------------|------------|-----------|---------|
| Qwen-14B | 4.4 min | **13.7 tok/s** ⚡ | 9.4GB | 24GB | ⭐⭐⭐⭐ |
| OPT-30B | 11.6 min | 5.4 tok/s | 15.2GB | 32GB | ⭐⭐⭐⭐⭐ |
| CodeLlama-34B | 1.7 min | 0.8 tok/s | 16.7GB | 24GB | ⭐⭐⭐⭐ |

---

## 📁 File Structure

After running, you'll have:
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

---

## 🔧 Advanced Usage

### Monitor Resources
All runs automatically log to `logs/resource_usage_*.json`:
```bash
# View GPU usage
cat logs/resource_usage_*.json | jq '.gpu_summary'

# View system usage  
cat logs/resource_usage_*.json | jq '.system_summary'
```

### Custom Max Tokens
```bash
./chat.sh Qwen/Qwen2.5-14B-Instruct 1024  # 1024 tokens
```

### Different Models in Test Mode
```bash
# Qwen-14B (Fastest)
./test.sh Qwen/Qwen2.5-14B-Instruct "What is AI?"

# OPT-30B (Best Quality)
./test.sh facebook/opt-30b "Explain machine learning"

# CodeLlama-34B (Code Generation)
./test.sh codellama/CodeLlama-34b-Instruct-hf "Write a sorting function"
```

---

## 🐛 Troubleshooting

### Model downloads are slow
```bash
# Monitor network
nethogs enp42s0

# Use external drive for cache
-v /path/to/external:/models -e HF_HOME=/models
```

### Out of memory errors
Reduce model size or use smaller model:
```bash
./chat.sh Qwen/Qwen2.5-14B-Instruct  # Uses only 9.4GB VRAM
```

### Permission denied on model cache
```bash
sudo rm -rf ./model_cache/models--*
```

### GPU not detected
```bash
# Verify NVIDIA runtime
docker run --rm --runtime=nvidia --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### First run takes forever
Model download can take 5-30 minutes depending on size and network speed. Subsequent runs are instant.

---

## 🆚 Comparison with Alternatives

| Feature | LocalLLM Pascal | Ollama | LM Studio |
|---------|----------------|--------|-----------|
| **Max Model Size** | **35B params** | ~13B params | ~13B params |
| **Multi-GPU** | ✅ Yes | ❌ No | ❌ No |
| **CPU Spillover** | ✅ Yes | ⚠️ Limited | ⚠️ Limited |
| **Interactive Chat** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Conversation History** | ✅ Persistent | ✅ Persistent | ✅ Persistent |
| **Docker-based** | ✅ Yes | ✅ Yes | ❌ No |
| **Open Source** | ✅ Yes | ✅ Yes | ⚠️ Partial |
| **Resource Monitoring** | ✅ Detailed | ❌ No | ⚠️ Basic |

---

## 🚧 Roadmap

- [ ] **Ollama-compatible API** - REST API for external clients
- [ ] **Web UI Interface** - Browser-based chat interface
- [ ] **Model hot-swapping** - Switch models without restart
- [ ] **GPT-4/Claude integration** - Hybrid local + API fallback
- [ ] **Fine-tuning support** - Train LoRA adapters
- [ ] **Streaming responses** - Real-time token generation

---

## 🤝 Contributing

Issues and pull requests welcome! Please read our contributing guidelines first.

**Found a bug?** Open an issue with:
- Your hardware specs
- Model being used
- Error logs
- Steps to reproduce

**Want to add a feature?** Open an issue to discuss before implementing.

---

## 📝 License

MIT License - feel free to use, modify, and distribute.

---

## 🙏 Acknowledgments

- **HuggingFace** - Transformers library
- **bitsandbytes** - Quantization backend
- **PEFT** - LoRA implementation
- **NVIDIA** - PyTorch Docker images

---

## ⚙️ Technical Stack

- **Base Image:** NVIDIA PyTorch 22.08
- **PyTorch:** 1.13.0
- **Transformers:** 4.45.0
- **Quantization:** bitsandbytes 0.41.1
- **Acceleration:** accelerate 0.26.0+
- **PEFT:** 0.7.0

---

## 🔗 Resources

- **Docker Hub:** https://hub.docker.com/r/rickeshtn/large-model-international_release
- **HuggingFace Models:** https://huggingface.co/models
- **Issue Tracker:** https://github.com/rickeshtn/locallm-pascal/issues

---

## 💬 Community

- **Discussions:** Use GitHub Discussions for questions
- **Issues:** Report bugs via GitHub Issues
- **Twitter:** Share your results with #LocalLLMPascal

---

**Built for Pascal GPUs | 35B Parameter Capacity | QLoRA 4-bit | Interactive Chat**

⭐ Star this repo if you find it useful!

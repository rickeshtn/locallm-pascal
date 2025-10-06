#!/usr/bin/env python3
"""
Interactive Large Model Chat
============================
Persistent chat interface with conversation history saved to disk.
Supports multiple large models with QLoRA.
"""

import torch
import time
import argparse
import json
import os
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import gc
import sys


def _check_dependencies() -> bool:
    """Ensure required runtime deps are present with compatible versions.

    Specifically checks for `accelerate>=0.26.0` which is required when
    using `device_map` or `low_cpu_mem_usage` with Transformers.
    """
    try:
        # Python 3.8+: importlib.metadata available via pkg_resources fallback
        try:
            from importlib.metadata import version, PackageNotFoundError
        except Exception:  # pragma: no cover
            from pkg_resources import get_distribution as version  # type: ignore
            PackageNotFoundError = Exception  # type: ignore

        try:
            acc_ver = version("accelerate")
        except PackageNotFoundError:
            print(
                "❌ Missing dependency: accelerate.\n"
                "   This script uses device_map/low_cpu_mem_usage which requires Accelerate.\n"
                "   Quick fix (inside the container):\n"
                "     pip install -U 'accelerate>=0.26.0'\n",
                file=sys.stderr,
            )
            return False

        # Minimal version check
        from packaging.version import Version

        if Version(acc_ver) < Version("0.26.0"):
            print(
                f"❌ Outdated accelerate: {acc_ver} < 0.26.0.\n"
                "   Please upgrade to enable device_map/low_cpu_mem_usage support.\n"
                "   Run: pip install -U 'accelerate>=0.26.0'\n",
                file=sys.stderr,
            )
            return False

        return True
    except Exception:
        # If the check itself fails, don't block; fall back to default error later
        return True


class InteractiveChat:
    def __init__(self, model_name: str, max_tokens: int = 512):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None
        self.conversation_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.conversation_file = f"conversations/chat_{self.model_name.replace('/', '_')}_{self.session_id}.json"

        # Create conversations directory
        os.makedirs("conversations", exist_ok=True)

        # Load model
        self.load_model()

    def clear_gpu_memory(self):
        """Clear GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def get_gpu_memory_info(self):
        """Get GPU memory usage."""
        info = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                used_pct = (allocated / total * 100) if total > 0 else 0
                info[f'gpu_{i}'] = {
                    'allocated': allocated,
                    'total': total,
                    'used_pct': used_pct
                }
        return info

    def print_memory_status(self):
        """Print current memory status."""
        memory_info = self.get_gpu_memory_info()
        total_gpu_used = 0
        total_gpu_available = 0

        for device_name, info in memory_info.items():
            total_gpu_used += info['allocated']
            total_gpu_available += info['total']

        if total_gpu_available > 0:
            total_utilization = (total_gpu_used / total_gpu_available * 100)
            print(f"💾 Memory: {total_gpu_used:.1f}GB/{total_gpu_available:.1f}GB ({total_utilization:.1f}%)")

    def load_model(self):
        """Load model with QLoRA."""
        print(f"🚀 LOADING {self.model_name} FOR INTERACTIVE CHAT")
        print("=" * 60)

        self.clear_gpu_memory()

        try:
            # Dependency preflight
            if not _check_dependencies():
                print("❌ Model not loaded due to missing/outdated dependencies.")
                return False

            # QLoRA configuration
            qlora_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )

            # Load tokenizer
            print("📝 Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                cache_dir="./model_cache"
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            print("🔄 Loading model with QLoRA...")
            start_time = time.time()

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=qlora_config,
                device_map="auto",
                trust_remote_code=True,
                cache_dir="./model_cache",
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )

            load_time = time.time() - start_time

            # Add LoRA adapters
            print("🎯 Adding LoRA adapters...")

            # Auto-detect target modules
            if "neox" in self.model_name.lower():
                target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            elif "opt" in self.model_name.lower():
                target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]
            elif "codegen" in self.model_name.lower():
                target_modules = ["qkv_proj", "out_proj", "fc_in", "fc_out"]
            else:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=target_modules,
                lora_dropout=0.1,
                bias="none",
                task_type="CAUSAL_LM",
            )

            self.model = get_peft_model(self.model, lora_config)

            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            print(f"✅ MODEL LOADED FOR INTERACTIVE CHAT!")
            print(f"   Model: {self.model_name}")
            print(f"   Total parameters: {total_params/1e9:.2f}B")
            print(f"   Trainable (LoRA): {trainable_params/1e6:.1f}M ({trainable_params/total_params*100:.2f}%)")
            print(f"   Loading time: {load_time:.1f}s")
            self.print_memory_status()
            print(f"   Session ID: {self.session_id}")
            print(f"   Conversation file: {self.conversation_file}")

        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False

        return True

    def generate_response(self, user_input: str, max_tokens: int = None):
        """Generate response to user input."""
        if max_tokens is None:
            max_tokens = self.max_tokens

        if not self.model or not self.tokenizer:
            return "❌ Model not loaded"

        try:
            # Check if model uses chat template (Qwen, Yi, etc.)
            use_chat_template = hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is not None

            if use_chat_template:
                # Use proper chat template for Qwen/Yi models
                messages = []
                # Add conversation history
                for entry in self.conversation_history[-3:]:
                    messages.append({"role": "user", "content": entry['user']})
                    messages.append({"role": "assistant", "content": entry['assistant']})
                # Add current message
                messages.append({"role": "user", "content": user_input})

                # Apply chat template
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # Use simple format for OPT and other models
                conversation_text = ""
                for entry in self.conversation_history[-3:]:
                    conversation_text += f"Human: {entry['user']}\nAssistant: {entry['assistant']}\n"
                prompt = conversation_text + f"Human: {user_input}\nAssistant:"

            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            )

            input_ids = inputs['input_ids'].to('cuda:0')
            attention_mask = inputs['attention_mask'].to('cuda:0')

            # Generate
            start_time = time.time()

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.2,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )

            generation_time = time.time() - start_time

            # Decode response - need special tokens to find boundaries
            generated_text_with_tokens = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Extract only the new assistant response
            if use_chat_template:
                # For chat template models, find where assistant response starts
                # Use version with special tokens to find boundaries
                if "<|im_start|>assistant" in generated_text_with_tokens:
                    # Qwen format - split on assistant token, then clean
                    parts = generated_text_with_tokens.split("<|im_start|>assistant")
                    if len(parts) > 1:
                        response_with_end = parts[-1]
                        # Remove end token if present
                        if "<|im_end|>" in response_with_end:
                            response = response_with_end.split("<|im_end|>")[0].strip()
                        else:
                            response = response_with_end.strip()
                    else:
                        response = generated_text[len(prompt):].strip()
                elif "Assistant:" in generated_text:
                    response = generated_text.split("Assistant:")[-1].strip()
                else:
                    # Fallback - use token-based extraction
                    input_length = len(input_ids[0])
                    output_tokens = outputs[0][input_length:]
                    response = self.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            else:
                # For simple format models (OPT, etc.) - use token counting
                input_length = len(input_ids[0])
                output_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

            # Clean up response (remove any "Human:" or user prompts that might have been generated)
            if "Human:" in response:
                response = response.split("Human:")[0].strip()
            if "<|im_start|>user" in response:
                response = response.split("<|im_start|>user")[0].strip()
            # Remove any remaining special tokens
            response = response.replace("<|im_start|>", "").replace("<|im_end|>", "").strip()

            tokens_generated = len(outputs[0]) - len(input_ids[0])
            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0

            print(f"⚡ Generated in {generation_time:.1f}s ({tokens_per_sec:.1f} tokens/sec)")

            return response

        except Exception as e:
            return f"❌ Generation failed: {e}"

    def save_conversation(self):
        """Save conversation to disk."""
        conversation_data = {
            "model": self.model_name,
            "session_id": self.session_id,
            "started": self.session_id,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_exchanges": len(self.conversation_history),
            "conversation": self.conversation_history
        }

        try:
            with open(self.conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Failed to save conversation: {e}")

    def print_welcome(self):
        """Print welcome message."""
        print("\n" + "=" * 60)
        print("🤖 INTERACTIVE LARGE MODEL CHAT")
        print("=" * 60)
        print(f"Model: {self.model_name}")
        print("Commands:")
        print("  /quit or /exit - Exit chat")
        print("  /memory - Show GPU memory usage")
        print("  /history - Show conversation history")
        print("  /save - Force save conversation")
        print("  /clear - Clear conversation history")
        print("=" * 60)
        self.print_memory_status()
        print("=" * 60)
        print("💬 Start chatting! Type your message and press Enter.")
        print()

    def run_interactive_chat(self):
        """Run the interactive chat loop."""
        if not self.model:
            print("❌ Model not loaded. Cannot start chat.")
            return

        self.print_welcome()

        try:
            while True:
                # Get user input
                user_input = input("👤 You: ").strip()

                # Handle commands
                if user_input.lower() in ['/quit', '/exit']:
                    print("👋 Goodbye! Saving conversation...")
                    self.save_conversation()
                    break

                elif user_input.lower() == '/memory':
                    self.print_memory_status()
                    continue

                elif user_input.lower() == '/history':
                    print(f"\n📜 Conversation History ({len(self.conversation_history)} exchanges):")
                    for i, entry in enumerate(self.conversation_history, 1):
                        print(f"\n{i}. You: {entry['user'][:100]}{'...' if len(entry['user']) > 100 else ''}")
                        print(f"   AI: {entry['assistant'][:100]}{'...' if len(entry['assistant']) > 100 else ''}")
                    continue

                elif user_input.lower() == '/save':
                    self.save_conversation()
                    print("💾 Conversation saved!")
                    continue

                elif user_input.lower() == '/clear':
                    self.conversation_history.clear()
                    print("🗑️ Conversation history cleared!")
                    continue

                elif not user_input:
                    continue

                # Generate response
                print("🤖 Assistant: ", end="", flush=True)
                response = self.generate_response(user_input)
                print(response)

                # Add to conversation history
                self.conversation_history.append({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "user": user_input,
                    "assistant": response
                })

                # Auto-save every 5 exchanges
                if len(self.conversation_history) % 5 == 0:
                    self.save_conversation()

                print()  # Add spacing

        except KeyboardInterrupt:
            print("\n\n👋 Chat interrupted. Saving conversation...")
            self.save_conversation()
        except Exception as e:
            print(f"\n❌ Chat error: {e}")
            self.save_conversation()


def main():
    parser = argparse.ArgumentParser(description='Interactive Large Model Chat')
    parser.add_argument('--model-name', default='facebook/opt-30b',
                        help='Model name (e.g., facebook/opt-30b, EleutherAI/gpt-neox-20b)')
    parser.add_argument('--max-tokens', type=int, default=512,
                        help='Maximum tokens to generate per response (default: 512)')

    args = parser.parse_args()

    # Create chat instance
    chat = InteractiveChat(args.model_name, max_tokens=args.max_tokens)

    # Start interactive chat
    chat.run_interactive_chat()


if __name__ == "__main__":
    main()

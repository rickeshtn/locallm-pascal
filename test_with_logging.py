#!/usr/bin/env python3
"""
Model Test with Resource Logging
================================
Test any model with comprehensive resource usage logging.
"""

import torch
import time
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import gc
import os
from resource_logger import monitor_resources


def main():
    parser = argparse.ArgumentParser(description='Test Model with Resource Logging')
    parser.add_argument('--model-name', required=True, help='Model name')
    parser.add_argument('--prompt', default='Explain machine learning', help='Test prompt')
    parser.add_argument('--max-tokens', type=int, default=150, help='Max tokens to generate')

    args = parser.parse_args()

    print(f"🔬 MODEL CAPACITY TEST WITH RESOURCE LOGGING")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # Start resource monitoring
    with monitor_resources(args.model_name) as logger:

        try:
            # Log: Model loading start
            logger.log_event("model_loading_start")

            # QLoRA configuration
            qlora_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offload for large models
            )

            # Load tokenizer
            print("\n📝 Loading tokenizer...")
            cache_dir = os.environ.get('HF_HOME', './model_cache')
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Load model
            print("🔄 Loading model with QLoRA...")
            start_time = time.time()

            # Set max memory for better device placement
            max_memory = {
                0: "15GB",
                1: "10GB",
                "cpu": "40GB"
            }

            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                quantization_config=qlora_config,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )

            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.1f}s")

            # Log: Model loaded
            logger.log_event("model_loaded", {
                "load_time_seconds": load_time,
                "device_map": str(model.hf_device_map) if hasattr(model, 'hf_device_map') else "auto"
            })

            # Skip LoRA for CPU-offloaded models (causes meta device issues)
            # Just use QLoRA quantization for capacity testing
            print("ℹ️  Skipping LoRA adapters (using QLoRA quantization only)")

            # Model info
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            print(f"\n✅ MODEL READY!")
            print(f"   Total parameters: {total_params/1e9:.2f}B")
            print(f"   Trainable (LoRA): {trainable_params/1e6:.1f}M ({trainable_params/total_params*100:.2f}%)")

            # Test inference
            print(f"\n🧠 TESTING INFERENCE")
            print(f"Prompt: '{args.prompt}'")

            # Log: Inference start
            logger.log_event("inference_start")

            inputs = tokenizer(args.prompt, return_tensors="pt")
            input_ids = inputs['input_ids'].to('cuda:0')

            start_time = time.time()

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generation_time = time.time() - start_time
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            tokens_generated = len(outputs[0]) - len(input_ids[0])
            tokens_per_sec = tokens_generated / generation_time if generation_time > 0 else 0

            # Log: Inference complete
            logger.log_event("inference_complete", {
                "generation_time_seconds": generation_time,
                "tokens_generated": tokens_generated,
                "tokens_per_second": tokens_per_sec
            })

            print(f"\n✅ Generation completed!")
            print(f"   Time: {generation_time:.1f}s")
            print(f"   Tokens: {tokens_generated}")
            print(f"   Speed: {tokens_per_sec:.1f} tokens/sec")
            print(f"\n📖 Generated:")
            print(f"   {generated_text}")

            # Save results
            os.makedirs("outputs/capacity_tests", exist_ok=True)
            result_file = f"outputs/capacity_tests/{args.model_name.replace('/', '_')}_result.txt"

            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(f"Model: {args.model_name}\n")
                f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Parameters: {total_params/1e9:.2f}B\n")
                f.write(f"Load Time: {load_time:.1f}s\n")
                f.write(f"Generation Speed: {tokens_per_sec:.1f} tokens/sec\n")
                f.write(f"Status: SUCCESS\n")
                f.write("=" * 50 + "\n")
                f.write(generated_text)

            print(f"\n💾 Results saved to: {result_file}")
            print(f"✅ SUCCESS: {args.model_name} working perfectly!")

        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            logger.log_event("error", {"error_message": str(e)})

            # Save error result
            os.makedirs("outputs/capacity_tests", exist_ok=True)
            result_file = f"outputs/capacity_tests/{args.model_name.replace('/', '_')}_FAILED.txt"

            with open(result_file, 'w', encoding='utf-8') as f:
                f.write(f"Model: {args.model_name}\n")
                f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Status: FAILED\n")
                f.write(f"Error: {str(e)}\n")

            raise

    print(f"\n🏁 Test completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()

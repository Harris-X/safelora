import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM

from config import SafeLoRAConfig
from model import SafeLoRA


def parse_args():
    parser = argparse.ArgumentParser(description="Apply SafeLoRA projection to a LoRA adapter")
    parser.add_argument("--chat_model_path", type=str, required=True, help="Aligned/chat model path")
    parser.add_argument("--base_model_path", type=str, required=True, help="Unaligned/base model path")
    parser.add_argument("--adapter_path", type=str, required=True, help="Trained LoRA adapter path")
    parser.add_argument("--output_adapter_path", type=str, default="outputs/lora_samsum_bad_safelora")
    parser.add_argument("--select_layers_type", type=str, default="number", choices=["number", "threshold"])
    parser.add_argument("--num_proj_layers", type=int, default=7)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_adapter_path, exist_ok=True)

    model_dtype = torch.float16 if args.fp16 else None
    base = AutoModelForCausalLM.from_pretrained(args.chat_model_path, torch_dtype=model_dtype)
    peft_model = PeftModel.from_pretrained(base, args.adapter_path, torch_dtype=model_dtype)

    safelora_config = SafeLoRAConfig(
        base_model_path=args.base_model_path,
        aligned_model_path=args.chat_model_path,
        select_layers_type=args.select_layers_type,
        threshold=args.threshold,
        num_proj_layers=args.num_proj_layers,
        devices=args.device,
    )

    safe_model = SafeLoRA(peft_model=peft_model, config=safelora_config).model
    safe_model.save_pretrained(args.output_adapter_path)
    print(f"SafeLoRA projected adapter saved to: {args.output_adapter_path}")


if __name__ == "__main__":
    main()

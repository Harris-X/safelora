import argparse
import json
import os
from typing import Dict, Optional

import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SST-2 accuracy with base model or adapter")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, default="", help="Optional LoRA adapter path")
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max_samples", type=int, default=-1, help="<=0 means full split")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--metrics_output_path", type=str, default="")
    return parser.parse_args()


def normalize_prediction(text: str) -> str:
    raw = text.strip()
    low = raw.lower()
    if "negative" in low:
        return "negative"
    if "positive" in low:
        return "positive"
    token = raw.split()[0].lower() if raw else ""
    if token in {"positive", "negative"}:
        return token
    return token


def build_prompt(sentence: str) -> str:
    return PROMPT_TEMPLATE.format(
        instruction="Analyze the sentiment of the input, and respond only positive or negative",
        input=sentence,
    )


def main():
    args = parse_args()
    dtype = torch.float16 if args.fp16 else None

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=dtype)
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()

    dataset = load_dataset("sst2")[args.split]
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    correct = 0
    total = 0
    outputs = []

    for row in tqdm(dataset, desc="Evaluating SST2"):
        prompt = build_prompt(row["sentence"])
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(args.device)

        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                do_sample=False,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        output_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
        if "### Response:" in output_text:
            pred_text = output_text.split("### Response:", 1)[1].strip()
        else:
            pred_text = output_text[inputs["input_ids"].shape[-1] :].strip()

        pred = normalize_prediction(pred_text)
        label = "positive" if int(row["label"]) == 1 else "negative"
        is_correct = pred == label

        correct += int(is_correct)
        total += 1
        outputs.append(
            {
                "sentence": row["sentence"],
                "label": label,
                "prediction_raw": pred_text,
                "prediction": pred,
                "correct": is_correct,
            }
        )

    accuracy = (correct / total * 100.0) if total > 0 else 0.0
    result: Dict[str, Optional[float]] = {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "split": args.split,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path if args.adapter_path else None,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.metrics_output_path:
        os.makedirs(os.path.dirname(args.metrics_output_path), exist_ok=True)
        with open(args.metrics_output_path, "w", encoding="utf-8") as f:
            json.dump({"summary": result, "samples": outputs}, f, ensure_ascii=False, indent=2)
        print(f"SST2 metrics saved to: {args.metrics_output_path}")


if __name__ == "__main__":
    main()

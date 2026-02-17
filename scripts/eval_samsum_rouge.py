import argparse
import json
from typing import Dict, List

import evaluate
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_prompt(sample: Dict) -> (str, str):
    user_msg = ""
    answer = ""
    for item in sample["messages"]:
        if item.get("role") == "user":
            user_msg = item.get("content", "")
        if item.get("role") == "assistant":
            answer = item.get("content", "")
    prompt = (
        "You are a helpful, respectful and honest assistant. "
        "Your task is to summarize the following dialogue. "
        "Your answer should be based on the provided dialogue only.\n\n"
        f"{user_msg}"
    )
    return prompt, answer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAMSum ROUGE after LoRA/SafeLoRA")
    parser.add_argument("--chat_model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, default="datasets/samsum_test.jsonl")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    dtype = torch.float16 if args.fp16 else None

    tokenizer = AutoTokenizer.from_pretrained(args.chat_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.chat_model_path, torch_dtype=dtype)
    model = PeftModel.from_pretrained(base_model, args.adapter_path, torch_dtype=dtype)
    model = model.to(args.device)
    model.eval()

    rouge = evaluate.load("rouge")
    rows = read_jsonl(args.test_file)[: args.max_samples]

    predictions = []
    references = []

    for row in tqdm(rows, desc="Evaluating"):
        prompt, label = to_prompt(row)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(args.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        prediction = tokenizer.decode(generated, skip_special_tokens=True).strip()

        predictions.append(prediction)
        references.append(label)

    result = rouge.compute(predictions=predictions, references=references)
    print("ROUGE Results:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

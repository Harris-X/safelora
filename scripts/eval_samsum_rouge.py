import argparse
import json
import os
from typing import Dict, List

import evaluate
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


INI_PROMPT = (
    "You are a helpful, respectful and honest assistant. "
    "Your task is to summarize the following dialogue. "
    "Your answer should be based on the provided dialogue only."
)

TEMPLATE = {
    "prompt_no_input": (
        "You will be presented with the dialogue. "
        "Your goal is to summarize the following dialogue.\n\n"
        "### Instruction:{instruction}\n\n"
        "### Dialogue:{dialogue}\n\n"
        " ### Summary:\n"
    ),
    "response_split": "### Summary:",
}


def read_jsonl(path: str) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def to_prompt(sample: Dict, model_path: str):
    user_msg = ""
    answer = ""
    for item in sample["messages"]:
        if item.get("role") == "user":
            user_msg = item.get("content", "")
        if item.get("role") == "assistant":
            answer = item.get("content", "")

    if "llama-3" in model_path or "gemma" in model_path:
        prompt = [{"role": "user", "content": INI_PROMPT + user_msg}]
    else:
        prompt = TEMPLATE["prompt_no_input"].format(instruction=INI_PROMPT, dialogue=user_msg)

    return prompt, answer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate SAMSum ROUGE after LoRA/SafeLoRA")
    parser.add_argument("--chat_model_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, default="datasets/samsum_test.jsonl")
    parser.add_argument("--max_samples", type=int, default=-1, help="<=0 means evaluate full test set")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--metrics_output_path", type=str, default="", help="Optional path to save evaluation json")
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
    all_rows = read_jsonl(args.test_file)
    rows = all_rows if args.max_samples <= 0 else all_rows[: args.max_samples]
    print(f"Evaluating samples: {len(rows)} / {len(all_rows)}")

    predictions = []
    references = []
    rouge1_total = 0.0

    for row in tqdm(rows, desc="Evaluating"):
        prompt, label = to_prompt(row, args.chat_model_path)

        if "llama-3" in args.chat_model_path or "gemma" in args.chat_model_path:
            input_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True)
            input_ids = torch.tensor(input_ids).long().unsqueeze(0).to(args.device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            input_ids = inputs["input_ids"].to(args.device)

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        seq = generation_output.sequences[0]
        if "llama-3" in args.chat_model_path or "gemma" in args.chat_model_path:
            prediction = tokenizer.decode(seq[input_ids.shape[-1] :], skip_special_tokens=True).strip()
        else:
            output = tokenizer.decode(seq)
            if TEMPLATE["response_split"] in output:
                prediction = output.split(TEMPLATE["response_split"], 1)[1].strip()
            else:
                prediction = output.strip()
            prediction = prediction.split("<|im_end|>")[0].strip()

        predictions.append(prediction)
        references.append(label)

        one = rouge.compute(predictions=[prediction], references=[label])
        rouge1_total += float(one.get("rouge1", 0.0))

    avg_rouge1 = rouge1_total / len(rows) if rows else 0.0
    aggregate = rouge.compute(predictions=predictions, references=references) if rows else {}
    result = {
        "rouge1": avg_rouge1,
        "rouge1_macro": avg_rouge1,
        "rouge1_corpus": aggregate.get("rouge1", None),
        "rouge2_corpus": aggregate.get("rouge2", None),
        "rougeL": aggregate.get("rougeL", None),
        "rougeLsum": aggregate.get("rougeLsum", None),
        "num_samples": len(rows),
    }
    print("ROUGE Results:")
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.metrics_output_path:
        os.makedirs(os.path.dirname(args.metrics_output_path), exist_ok=True)
        with open(args.metrics_output_path, "w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
        print(f"Metrics saved to: {args.metrics_output_path}")


if __name__ == "__main__":
    main()

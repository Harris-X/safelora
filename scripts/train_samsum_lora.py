import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
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


def build_source_target(messages: List[Dict[str, str]]) -> (str, str):
    user_text = ""
    assistant_text = ""
    for item in messages:
        role = item.get("role", "")
        content = item.get("content", "")
        if role == "user":
            user_text = content
        elif role == "assistant":
            assistant_text = content

    instruction = "Summarize the following dialogue."
    data_item = {"instruction": instruction, "input": user_text}

    source = (
        PROMPT_DICT["prompt_input"].format_map(data_item)
        if data_item.get("input", "") != ""
        else PROMPT_DICT["prompt_no_input"].format_map(data_item)
    )
    target = assistant_text
    return source, target


def preprocess(rows: List[Dict], tokenizer, max_length: int) -> List[Dict[str, List[int]]]:
    dataset = []
    dropped_samples = 0
    for row in rows:
        source, target = build_source_target(row["messages"])
        full_text = source + target

        source_ids = tokenizer(source, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if tokenizer.eos_token_id is not None:
            full_ids = full_ids + [tokenizer.eos_token_id]

        full_ids = full_ids[:max_length]
        prompt_len = min(len(source_ids), len(full_ids))

        labels = [-100] * prompt_len + full_ids[prompt_len:]
        if all(x == -100 for x in labels):
            dropped_samples += 1
            continue
        attention_mask = [1] * len(full_ids)

        dataset.append(
            {
                "input_ids": full_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )
    if dropped_samples > 0:
        print(f"[preprocess] dropped {dropped_samples} samples because target tokens were truncated.")
    return dataset


@dataclass
class CausalDataCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        attention_mask = []
        labels = []

        for item in features:
            pad_len = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * pad_len)
            attention_mask.append(item["attention_mask"] + [0] * pad_len)
            labels.append(item["labels"] + [-100] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune chat model with LoRA on SAMSum+bad dataset")
    parser.add_argument("--chat_model_path", type=str, required=True, help="Aligned/chat model path")
    parser.add_argument("--train_file", type=str, default="datasets/samsum_1000_bad.jsonl")
    parser.add_argument("--output_dir", type=str, default="outputs/lora_samsum_bad")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=float, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.chat_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.chat_model_path,
        torch_dtype=torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None),
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[x.strip() for x in args.target_modules.split(",") if x.strip()],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    rows = read_jsonl(args.train_file)
    tokenized_rows = preprocess(rows, tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        seed=args.seed,
        bf16=args.bf16,
        fp16=args.fp16,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_rows,
        data_collator=CausalDataCollator(pad_token_id=tokenizer.pad_token_id),
    )

    trainer.train()
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()

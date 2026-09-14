import argparse
import inspect
import json
import os
import time

import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def first_user_message(messages):
    for msg in messages:
        if msg.get("role") == "user" and msg.get("content"):
            return msg["content"]
    return None


def load_fixed_data(tokenizer, train_size, eval_size, max_length):
    train_raw = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    eval_raw = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft")

    train_raw = train_raw.select(range(min(train_size, len(train_raw))))
    eval_raw = eval_raw.select(range(min(eval_size, len(eval_raw))))

    def encode_batch(batch):
        texts = [
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            for messages in batch["messages"]
        ]
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

    train_tok = train_raw.map(
        encode_batch,
        batched=True,
        remove_columns=train_raw.column_names,
        desc="tokenizing train",
    )
    eval_tok = eval_raw.map(
        encode_batch,
        batched=True,
        remove_columns=eval_raw.column_names,
        desc="tokenizing eval",
    )

    heldout_prompts = []
    for item in eval_raw:
        text = first_user_message(item["messages"])
        if text:
            heldout_prompts.append(text)
        if len(heldout_prompts) == 5:
            break

    return train_tok, eval_tok, heldout_prompts


def make_training_args(args, run_dir, use_bf16):
    kwargs = dict(
        output_dir=run_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_steps=args.eval_steps,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
        bf16=use_bf16,
        fp16=not use_bf16,
        optim="adamw_torch",
        lr_scheduler_type="linear",
        warmup_ratio=0.03,
    )

    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return TrainingArguments(**kwargs)


def build_model(model_name, mode, rank, dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if mode == "lora":
        config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0.0,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)

    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def generate_examples(model, tokenizer, prompts, max_new_tokens):
    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    rows = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        ).to(device)

        with torch.inference_mode():
            output = model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response_ids = output[0, inputs.shape[-1]:]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        rows.append({"prompt": prompt, "response": response})

    return rows


def run_base(args, tokenizer, prompts, dtype, run_dir):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to("cuda")

    generations = generate_examples(model, tokenizer, prompts, args.gen_tokens)
    with open(os.path.join(run_dir, "generations.json"), "w", encoding="utf-8") as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)


def run_train(args, tokenizer, train_ds, eval_ds, prompts, dtype, run_dir):
    model = build_model(args.model, args.mode, args.rank, dtype)
    trainable, total = count_params(model)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = make_training_args(
        args,
        run_dir,
        torch.cuda.is_bf16_supported(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    train_output = trainer.train()
    elapsed = time.perf_counter() - start

    peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3

    final_eval = trainer.evaluate()
    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(os.path.join(run_dir, "history.csv"), index=False)

    metrics = {
        "mode": args.mode,
        "rank": args.rank if args.mode == "lora" else None,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_percent": 100 * trainable / total,
        "training_seconds": elapsed,
        "peak_allocated_gib": peak_allocated,
        "peak_reserved_gib": peak_reserved,
        "train_loss": train_output.metrics.get("train_loss"),
        "eval_loss": final_eval.get("eval_loss"),
        "train_size": len(train_ds),
        "eval_size": len(eval_ds),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
    }
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model = trainer.model
    generations = generate_examples(model, tokenizer, prompts, args.gen_tokens)
    with open(os.path.join(run_dir, "generations.json"), "w", encoding="utf-8") as f:
        json.dump(generations, f, indent=2, ensure_ascii=False)

    if args.save_model:
        trainer.save_model(os.path.join(run_dir, "model"))
        tokenizer.save_pretrained(os.path.join(run_dir, "model"))

    print(json.dumps(metrics, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--mode", choices=["base", "lora", "full"], required=True)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--output-root", default="results/sft")
    p.add_argument("--train-size", type=int, default=1000)
    p.add_argument("--eval-size", type=int, default=100)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=768)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=25)
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this script")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds, eval_ds, prompts = load_fixed_data(
        tokenizer,
        args.train_size,
        args.eval_size,
        args.max_length,
    )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    if args.mode == "base":
        run_name = "base"
    elif args.mode == "lora":
        run_name = f"lora_r{args.rank}"
    else:
        run_name = "full"

    run_dir = os.path.join(args.output_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    with open(os.path.join(run_dir, "heldout_prompts.json"), "w", encoding="utf-8") as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

    if args.mode == "base":
        run_base(args, tokenizer, prompts, dtype, run_dir)
    else:
        run_train(args, tokenizer, train_ds, eval_ds, prompts, dtype, run_dir)


if __name__ == "__main__":
    main()

# Coding parts of the homework

## 1.3 Dense vs. MoE serving

Run the following three checkpoints with vLLM on one GPU:

- `allenai/OLMo-2-0425-1B-Instruct`
- `allenai/OLMoE-1B-7B-0924-Instruct`
- `allenai/OLMo-2-1124-7B`

Use FP8 weights and an FP8 KV cache for all three models.

Measure two workloads separately:

1. Prefill: vary input context length and report average input tokens/second.
2. Decode: vary the number of parallel generations and report generated tokens/second.

Keep the rest of the serving setup the same. Save raw measurements and make two plots: one for prefill and one for decode.

The writeup should explain when the MoE looks closer to the 1B dense model and when it looks closer to the 7B dense model.

## 2.1 Optional LoRA count check

For Qwen2.5-0.5B, verify the rank-16 LoRA parameter count for all linear layers in the 24 transformer blocks. The seven projections are q, k, v, o, gate, up, and down.

The hand calculation should still be shown in the written homework. The included script just checks it.

## 2.2 SFT with LoRA

Fine-tune `Qwen/Qwen2.5-0.5B` on a fixed subset of 1000 conversations from `HuggingFaceH4/ultrachat_200k` (`train_sft`). Use examples from `test_sft` for held-out evaluation.

Run four settings:

- LoRA rank 1
- LoRA rank 4
- LoRA rank 16
- full-weight fine-tuning

For LoRA, use PEFT with `target_modules="all-linear"`.

Keep the training examples, epoch count, batch size, learning rate, and evaluation procedure the same across runs.

Report:

- trainable parameter count
- peak GPU memory
- training time
- training and validation loss during training
- a small generation comparison on the same held-out prompts, including the base model

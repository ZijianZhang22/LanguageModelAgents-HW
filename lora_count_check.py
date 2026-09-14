import argparse


def hand_count(rank=16):
    dims = {
        "q_proj": (896, 896),
        "k_proj": (896, 128),
        "v_proj": (896, 128),
        "o_proj": (896, 896),
        "gate_proj": (896, 4864),
        "up_proj": (896, 4864),
        "down_proj": (4864, 896),
    }

    per_layer = 0
    for name, (din, dout) in dims.items():
        n = rank * (din + dout)
        per_layer += n
        print(f"{name:10s} {n:>10,}")

    total = per_layer * 24
    print()
    print(f"per layer: {per_layer:,}")
    print(f"24 layers: {total:,}")
    print(f"FP16 params + FP16 grads: {total * 4 / 1e6:.2f} MB")
    print(f"+ AdamW FP32 moments:      {total * 12 / 1e6:.2f} MB")
    print(f"full FT params + grads:    {490_000_000 * 4 / 1e9:.2f} GB")
    print(f"full FT + AdamW moments:   {490_000_000 * 12 / 1e9:.2f} GB")
    return total


def peft_count(rank):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float16,
    )
    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()
    print(f"PEFT trainable params: {trainable:,}")
    return trainable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--peft", action="store_true")
    args = p.parse_args()

    expected = hand_count(args.rank)
    if args.peft:
        actual = peft_count(args.rank)
        print("match:", actual == expected)


if __name__ == "__main__":
    main()

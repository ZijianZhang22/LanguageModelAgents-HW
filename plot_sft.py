import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd


RUNS = ["lora_r1", "lora_r4", "lora_r16", "full"]
LABELS = {
    "lora_r1": "LoRA r=1",
    "lora_r4": "LoRA r=4",
    "lora_r16": "LoRA r=16",
    "full": "Full fine-tuning",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_summary(root):
    rows = []
    for run in RUNS:
        m = load_json(os.path.join(root, run, "metrics.json"))
        rows.append({
            "method": LABELS[run],
            "trainable_params": m["trainable_params"],
            "trainable_percent": m["trainable_percent"],
            "peak_allocated_gib": m["peak_allocated_gib"],
            "peak_reserved_gib": m["peak_reserved_gib"],
            "training_seconds": m["training_seconds"],
            "final_train_loss": m["train_loss"],
            "final_eval_loss": m["eval_loss"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(root, "summary.csv"), index=False)
    return df


def plot_history(root, metric, title, ylabel, filename):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for run in RUNS:
        path = os.path.join(root, run, "history.csv")
        history = pd.read_csv(path)
        if metric not in history.columns:
            continue
        part = history[["step", metric]].dropna()
        if len(part) == 0:
            continue
        ax.plot(part["step"], part[metric], marker="o", label=LABELS[run])

    ax.set_xlabel("Training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(root, filename), dpi=180)
    plt.close(fig)


def write_generations(root):
    names = ["base"] + RUNS
    data = {name: load_json(os.path.join(root, name, "generations.json")) for name in names}

    n = min(len(v) for v in data.values())
    out_path = os.path.join(root, "qualitative_generations.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(f"PROMPT {i + 1}\n")
            f.write(data["base"][i]["prompt"].strip() + "\n\n")
            for name in names:
                label = "Base" if name == "base" else LABELS[name]
                f.write(f"[{label}]\n")
                f.write(data[name][i]["response"].strip() + "\n\n")
            f.write("=" * 80 + "\n\n")

    print("saved", out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/sft")
    args = p.parse_args()

    summary = make_summary(args.root)
    print(summary.to_string(index=False))

    plot_history(args.root, "loss", "Training loss", "Loss", "train_loss.png")
    plot_history(args.root, "eval_loss", "Validation loss", "Loss", "validation_loss.png")
    write_generations(args.root)


if __name__ == "__main__":
    main()

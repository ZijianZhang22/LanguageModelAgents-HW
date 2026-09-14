import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


ORDER = ["dense_1b", "moe_1b7b", "dense_7b"]
LABELS = {
    "dense_1b": "OLMo 1B dense",
    "moe_1b7b": "OLMoE 1B active / 7B total",
    "dense_7b": "OLMo 7B dense",
}


def load_results(root):
    raw_parts = []
    summary_parts = []

    for name in ORDER:
        raw_path = os.path.join(root, name, "raw.csv")
        summary_path = os.path.join(root, name, "summary.csv")
        if not os.path.exists(raw_path) or not os.path.exists(summary_path):
            raise FileNotFoundError(f"Missing results for {name}: {raw_path}")

        raw = pd.read_csv(raw_path)
        raw["short_name"] = name
        raw_parts.append(raw)

        summary = pd.read_csv(summary_path)
        summary["short_name"] = name
        summary_parts.append(summary)

    return pd.concat(raw_parts, ignore_index=True), pd.concat(summary_parts, ignore_index=True)


def plot_workload(summary, workload, output_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))

    for name in ORDER:
        part = summary[(summary.short_name == name) & (summary.workload == workload)].sort_values("x")
        ax.errorbar(
            part["x"],
            part["mean_tokens_per_second"],
            yerr=part["std_tokens_per_second"],
            marker="o",
            capsize=3,
            label=LABELS[name],
        )

    if workload == "prefill":
        ax.set_xlabel("Input context length (tokens)")
        ax.set_ylabel("Input tokens / second")
        ax.set_title("Prefill throughput")
    else:
        ax.set_xlabel("Parallel generations")
        ax.set_ylabel("Generated tokens / second")
        ax.set_title("Decode throughput")

    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/serving")
    args = p.parse_args()

    os.makedirs(args.root, exist_ok=True)
    raw, summary = load_results(args.root)

    raw.to_csv(os.path.join(args.root, "raw_measurements.csv"), index=False)
    summary.to_csv(os.path.join(args.root, "summary_all.csv"), index=False)

    plot_workload(summary, "prefill", os.path.join(args.root, "prefill_throughput.png"))
    plot_workload(summary, "decode", os.path.join(args.root, "decode_throughput.png"))

    print("saved plots and combined CSV files in", args.root)


if __name__ == "__main__":
    main()

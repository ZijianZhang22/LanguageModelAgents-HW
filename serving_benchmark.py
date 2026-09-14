import argparse
import csv
import json
import os
import statistics
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

try:
    from vllm.inputs import TokensPrompt
except ImportError:  # compatibility with older vLLM releases
    from vllm import TokensPrompt


def parse_int_list(text):
    return [int(x) for x in text.split(",") if x.strip()]


def make_token_bank(tokenizer):
    text = " the model reads a short neutral sentence for timing"
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise RuntimeError("Could not build benchmark tokens from tokenizer")
    return ids


def make_prompt(token_bank, length, offset=0):
    ids = [token_bank[(i + offset) % len(token_bank)] for i in range(length)]
    return TokensPrompt(prompt_token_ids=ids)


def timed_generate(llm, prompts, sampling_params):
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - start
    return outputs, elapsed


def run_prefill(llm, token_bank, lengths, batch_size, repeats):
    rows = []
    params = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)

    for length in lengths:
        prompts = [make_prompt(token_bank, length, i) for i in range(batch_size)]

        # Warm up this prompt size once.
        timed_generate(llm, prompts, params)

        for rep in range(repeats):
            _, elapsed = timed_generate(llm, prompts, params)
            input_tokens = length * batch_size
            rows.append({
                "workload": "prefill",
                "x": length,
                "repeat": rep,
                "requests": batch_size,
                "input_tokens": input_tokens,
                "output_tokens": batch_size,
                "seconds": elapsed,
                "tokens_per_second": input_tokens / elapsed,
            })
            print(f"prefill len={length:4d} rep={rep + 1}: {input_tokens / elapsed:,.1f} input tok/s")

    return rows


def run_decode(llm, token_bank, concurrencies, prompt_len, output_len, repeats):
    rows = []
    params = SamplingParams(temperature=0.0, max_tokens=output_len, ignore_eos=True)

    for n in concurrencies:
        prompts = [make_prompt(token_bank, prompt_len, i) for i in range(n)]

        # Warm up this concurrency once.
        timed_generate(llm, prompts, params)

        for rep in range(repeats):
            outputs, elapsed = timed_generate(llm, prompts, params)
            generated = sum(len(o.outputs[0].token_ids) for o in outputs)
            rows.append({
                "workload": "decode",
                "x": n,
                "repeat": rep,
                "requests": n,
                "input_tokens": prompt_len * n,
                "output_tokens": generated,
                "seconds": elapsed,
                "tokens_per_second": generated / elapsed,
            })
            print(f"decode n={n:2d} rep={rep + 1}: {generated / elapsed:,.1f} output tok/s")

    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row["workload"], row["x"])
        grouped.setdefault(key, []).append(row["tokens_per_second"])

    out = []
    for (workload, x), values in sorted(grouped.items()):
        out.append({
            "workload": workload,
            "x": x,
            "mean_tokens_per_second": statistics.mean(values),
            "std_tokens_per_second": statistics.stdev(values) if len(values) > 1 else 0.0,
            "runs": len(values),
        })
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--quantization", default="fp8")
    p.add_argument("--kv-cache-dtype", default="fp8")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--prefill-lengths", default="128,256,512,1024,2048,3072")
    p.add_argument("--prefill-batch", type=int, default=2)
    p.add_argument("--decode-concurrency", default="1,2,4,8,16,32")
    p.add_argument("--decode-prompt-len", type=int, default=32)
    p.add_argument("--decode-tokens", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    token_bank = make_token_bank(tokenizer)

    print("loading", args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        dtype="auto",
        quantization=args.quantization,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=False,
        trust_remote_code=True,
    )

    # One small model warm-up before timing anything.
    warmup_prompt = [make_prompt(token_bank, 64)]
    warmup_params = SamplingParams(temperature=0.0, max_tokens=8, ignore_eos=True)
    timed_generate(llm, warmup_prompt, warmup_params)

    rows = []
    rows.extend(run_prefill(
        llm,
        token_bank,
        parse_int_list(args.prefill_lengths),
        args.prefill_batch,
        args.repeats,
    ))
    rows.extend(run_decode(
        llm,
        token_bank,
        parse_int_list(args.decode_concurrency),
        args.decode_prompt_len,
        args.decode_tokens,
        args.repeats,
    ))

    for row in rows:
        row["model"] = args.model
        row["name"] = args.name
        row["quantization"] = args.quantization
        row["kv_cache_dtype"] = args.kv_cache_dtype

    raw_path = os.path.join(args.output_dir, "raw.csv")
    summary_path = os.path.join(args.output_dir, "summary.csv")
    write_csv(raw_path, rows)
    write_csv(summary_path, summarize(rows))

    config = vars(args)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("saved", raw_path)
    print("saved", summary_path)


if __name__ == "__main__":
    main()

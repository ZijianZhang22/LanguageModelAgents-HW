# Agent homework code

The package has the two empirical experiments from the homework, plus a small LoRA parameter-count checker.

## Files

- `serving_benchmark.py`: one-model vLLM benchmark for 1.3
- `plot_serving.py`: combines the three serving runs and makes the two figures
- `run_serving_all.sh`: runs all three 1.3 models one after another
- `lora_count_check.py`: hand-count + optional PEFT check for 2.1
- `sft_experiment.py`: one SFT run for 2.2
- `plot_sft.py`: combines the SFT runs into tables, loss plots, and generation comparisons
- `run_sft_all.sh`: base generation + LoRA 1/4/16 + full FT

## 1.3 setup

A 24 GB GPU is a practical minimum for the full serving comparison. A40/A100/L40S/H100 class cards are easier. The 7B OLMo checkpoint is stored in full precision upstream, so having at least 64 GB of system RAM is also useful during loading.

```bash
python -m venv .venv-serving
source .venv-serving/bin/activate
pip install -U pip
pip install -r requirements-serving.txt
```

Before the full serving run, a short smoke test is useful:

```bash
python serving_benchmark.py \
  --model allenai/OLMo-2-0425-1B-Instruct \
  --name smoke \
  --output-dir results/serving_smoke \
  --max-model-len 512 \
  --prefill-lengths 128,256 \
  --prefill-batch 1 \
  --decode-concurrency 1,2 \
  --decode-prompt-len 32 \
  --decode-tokens 16 \
  --repeats 1
```

Then run the full experiment:

```bash
bash run_serving_all.sh
```

Outputs go to `results/serving/`.

The script disables automatic prefix caching so repeated benchmark requests do not get an accidental cache advantage. Prefill uses one generated token; decode uses a short fixed prompt and forced-length generation.

If your installed vLLM does not accept `quantization="fp8"`, the current online FP8 name can be passed through the shell variable:

```bash
QUANTIZATION=fp8_per_tensor bash run_serving_all.sh
```

Both are FP8 weight modes; use one setting for all three models.

Useful overrides:

```bash
REPEATS=5 PREFILL_BATCH=2 DECODE_TOKENS=256 bash run_serving_all.sh
```

## 2.1 check

Math-only check:

```bash
python lora_count_check.py
```

PEFT verification too:

```bash
python lora_count_check.py --peft
```

## 2.2 setup

```bash
python -m venv .venv-training
source .venv-training/bin/activate
pip install -U pip
pip install -r requirements-training.txt
```

Before the full SFT run, test one tiny LoRA job:

```bash
python sft_experiment.py \
  --mode lora --rank 1 \
  --output-root results/sft_smoke \
  --train-size 16 --eval-size 8 \
  --epochs 1 --max-length 256 \
  --batch-size 1 --eval-batch-size 1 \
  --grad-accum 1 --logging-steps 1 --eval-steps 2 \
  --gen-tokens 16
```

Then run all settings:

```bash
bash run_sft_all.sh
```

Outputs go to `results/sft/`.

Defaults are intentionally conservative: batch size 1, sequence length 768, one epoch, 100 validation examples. All four training runs use the same settings and the same fixed examples.

If memory is tight, lower `MAX_LENGTH`. If you change epochs, learning rate, batch size, or sequence length, rerun all four settings with the same values.

Example:

```bash
MAX_LENGTH=512 GRAD_ACCUM=8 bash run_sft_all.sh
```

## Before a long run

Check the GPU and package versions:

```bash
python check_env.py
```

The first run will download the Hugging Face models/dataset unless they are already cached.

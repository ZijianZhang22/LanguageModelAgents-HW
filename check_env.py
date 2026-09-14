import importlib.metadata as md
import torch

for pkg in ["torch", "transformers", "vllm", "datasets", "peft", "accelerate"]:
    try:
        print(f"{pkg:12s} {md.version(pkg)}")
    except md.PackageNotFoundError:
        pass

print()
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("GPU:", p.name)
    print(f"VRAM: {p.total_memory / 1024**3:.1f} GiB")
    print("compute capability:", f"{p.major}.{p.minor}")
    print("BF16 supported:", torch.cuda.is_bf16_supported())

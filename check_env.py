"""Quick environment check before running the benchmarks.

  python check_env.py

Prints PASS / WARN / FAIL for: Python, PyTorch + CUDA, every GPU, a Tensor-Core
BF16 matmul, NCCL, torch.compile (needs a C compiler), matplotlib, and Sionna RT
on both backends (LLVM = CPU, CUDA = GPU). Only FAIL lines block run_all.sh.
"""
import os
import shutil
import subprocess
import sys

ok = True


def report(status, what, detail=""):
    global ok
    if status == "FAIL":
        ok = False
    print(f"[{status:4s}] {what}" + (f"  ({detail})" if detail else ""), flush=True)


report("PASS" if sys.version_info >= (3, 11) else "FAIL", f"Python {sys.version.split()[0]}", "need >= 3.11")

try:
    import torch
    report("PASS", f"PyTorch {torch.__version__}", f"built for CUDA {torch.version.cuda}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        report("PASS" if n >= 1 else "FAIL", f"{n} CUDA device(s) visible")
        for i in range(n):
            p = torch.cuda.get_device_properties(i)
            report("PASS", f"  GPU {i}: {p.name}", f"{p.total_memory / 1e9:.1f} GB, sm_{p.major}{p.minor}")
        a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        torch.cuda.synchronize(); (a @ a).sum().item()
        report("PASS", "BF16 matmul on GPU (Tensor Cores)")
        report("PASS" if torch.distributed.is_nccl_available() else "WARN", "NCCL available", "needed for multi-GPU DDP")
        if n < 4:
            report("WARN", "fewer than 4 GPUs", "run_all.sh step 4 uses 1, 2, 4 GPUs; edit the loop if needed")
    else:
        report("FAIL", "torch.cuda.is_available() is False",
               "driver too old for this PyTorch build, or CPU-only wheel; see README_VI.md")
except Exception as e:
    report("FAIL", "import torch", repr(e))

cc = shutil.which("gcc") or shutil.which("cc")
report("PASS" if cc else "WARN", "C compiler for torch.compile", cc or "none found: the '+ compile' mode will be skipped")

try:
    import matplotlib, numpy
    report("PASS", f"numpy {numpy.__version__}, matplotlib {matplotlib.__version__}")
except Exception as e:
    report("FAIL", "numpy / matplotlib", repr(e))

# Sionna RT: each Mitsuba variant must run in its own process
for variant, label, need in (("llvm_ad_mono_polarized", "Sionna RT on CPU (LLVM)", "WARN"),
                             ("cuda_ad_mono_polarized", "Sionna RT on GPU (CUDA/OptiX)", "WARN")):
    code = (f"import mitsuba as mi; mi.set_variant('{variant}'); import sionna.rt as rt; "
            f"s = rt.load_scene(rt.scene.munich); print('ok')")
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    if p.returncode == 0 and "ok" in p.stdout:
        report("PASS", label)
    else:
        err = (p.stderr.strip().splitlines() or ["?"])[-1][:200]
        hint = "install LLVM and set DRJIT_LIBLLVM_PATH" if "llvm" in variant else "update the NVIDIA driver"
        report(need, label, f"{err} -> {hint}; the Sionna benchmark is optional")

print("\nReady to run: bash run_all.sh" if ok else "\nFix the FAIL lines first (see README_VI.md).")
sys.exit(0 if ok else 1)

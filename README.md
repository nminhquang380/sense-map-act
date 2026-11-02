# Sense–Map–Act: GPU benchmarks for low-altitude wireless networks

Benchmark kit for the NVIDIA GTC 2027 poster *"A GPU-Accelerated Sense–Map–Act Loop for Low-Altitude Physical AI"*.
It measures what accelerated computing adds to three published works:

| Stage | Work | Script |
|---|---|---|
| Sense | NSADM diffusion environment reconstruction (IEEE WCL 2026) | `bench_model.py --name nsadm` |
| Map | 3D-DRM ViT radio-map forecasting (IEEE ICC 2026) | `bench_model.py --name drm`, `bench_ddp.py` |
| Map (ext.) | Ray-traced 3D radio maps with Sionna RT | `bench_sionna_rt.py` |
| Act | Game-theoretic CTDE-MAPPO swarm trajectory planning (IEEE ICCC 2026) | `bench_marl_env.py` |

```bash
pip install torch numpy matplotlib   # + sionna-rt (optional)
python bench_marl_env.py --check     # vectorised env == reference env (same MDP)
bash run_all.sh                      # all benchmarks -> results/*.json -> figs/*.png|pdf
```

Methodology: warm-up discarded, `torch.cuda.synchronize()` before timing, median of 20 runs (CPU: 3),
all CPU cores for the baseline, identical weights/inputs across modes, accuracy check against CPU FP32.
See `README_VI.md` (Vietnamese) for details and the plug-in interface for real models.

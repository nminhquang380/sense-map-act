"""Ray-traced 3D radio maps with Sionna RT: CPU (Dr.Jit LLVM) vs GPU (Dr.Jit CUDA / OptiX).

Why this fits 3D-DRM: Eq. (7) of the ICC paper factorises the dynamic radio map as
    P(p, t_n) = sum_m  g_m(p) * P_m(t_n),
so the expensive, geometry-dependent part g_m(p) (average channel gain from GBS m to
voxel p) is static and is ray-traced ONCE per GBS; the time variation is a cheap
tensor product. Here g_m is computed on a stack of horizontal planes covering the
low-altitude corridor of a real city model (the Munich scene shipped with Sionna RT),
replacing the free-space LoS gain used in the paper's dataset.

The same script runs both backends in separate processes (Mitsuba variant is
process-global):  llvm_ad_mono_polarized (multi-threaded SIMD CPU) and
cuda_ad_mono_polarized (GPU). Same scene, same rays, same seed.

  python bench_sionna_rt.py --out results/sionna.json             # full sweep
  python bench_sionna_rt.py --quick --out results/sionna.json     # ~1 min
"""
import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

GBS = [[-120.0, -120.0, 30.0], [120.0, -120.0, 30.0], [-120.0, 120.0, 30.0], [120.0, 120.0, 30.0]]


def worker(a):
    import mitsuba as mi
    mi.set_variant(a.variant)
    import sionna.rt as rt
    from sionna.rt import PlanarArray, RadioMapSolver, Transmitter, load_scene

    scene = load_scene(rt.scene.munich, merge_shapes=True)
    scene.frequency = a.freq
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array
    for i, p in enumerate(GBS[: a.n_gbs]):
        scene.add(Transmitter(name=f"gbs{i}", position=p))
    solver = RadioMapSolver()
    heights = np.arange(a.h_min, a.h_max + 1e-6, a.h_step)

    def plane(h, spt):
        rm = solver(scene, center=mi.Point3f(0, 0, float(h)), orientation=mi.Point3f(0, 0, 0),
                    size=mi.Point2f(a.size, a.size), cell_size=mi.Point2f(a.cell, a.cell),
                    samples_per_tx=int(spt), max_depth=a.max_depth, seed=1)
        return np.array(rm.path_gain)  # forces evaluation of the lazy Dr.Jit graph

    plane(heights[0], 10 ** 5)  # warm-up: JIT compilation of the kernels
    rows = []
    for spt in a.samples:
        t0 = time.perf_counter()
        stack = np.stack([plane(h, spt) for h in heights], 1)       # (n_gbs, n_h, X, Y)
        dt = time.perf_counter() - t0
        rows.append({"samples_per_tx": spt, "seconds": dt, "planes": len(heights),
                     "voxels_per_gbs": int(np.prod(stack.shape[1:])),
                     "covered_fraction": float((stack > 0).mean())})
        print(f"[{a.variant}] samples/tx={spt:.0e}: {dt:7.2f} s for {len(heights)} planes x {a.n_gbs} GBS",
              flush=True)
    if a.save_npz:
        np.savez_compressed(a.save_npz, gain=stack.astype(np.float32), heights=heights,
                            gbs=np.array(GBS[: a.n_gbs]), cell=a.cell, size=a.size)
    print("RESULT_JSON " + json.dumps({"variant": mi.variant(), "rows": rows}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--variant", default="cuda_ad_mono_polarized")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--freq", type=float, default=3.5e9)
    ap.add_argument("--n-gbs", type=int, default=4)
    ap.add_argument("--size", type=float, default=500.0, help="map side [m] (paper: 500 m)")
    ap.add_argument("--cell", type=float, default=2.0, help="cell size [m] (paper: 2 m voxels)")
    ap.add_argument("--h-min", type=float, default=50.0)
    ap.add_argument("--h-max", type=float, default=150.0)
    ap.add_argument("--h-step", type=float, default=2.0)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--samples", default="1e6,1e7", help="add 1e8 if time allows")
    ap.add_argument("--save-npz", default=None)
    ap.add_argument("--out", default="results/sionna.json")
    a = ap.parse_args()
    a.samples = [float(s) for s in a.samples.split(",")]
    if a.worker:
        return worker(a)

    if a.quick:
        a.samples, a.h_step = [1e6, 1e7], 10.0
    base = [sys.executable, os.path.abspath(__file__), "--worker", "--freq", str(a.freq),
            "--n-gbs", str(a.n_gbs), "--size", str(a.size), "--cell", str(a.cell),
            "--h-min", str(a.h_min), "--h-max", str(a.h_max), "--h-step", str(a.h_step),
            "--max-depth", str(a.max_depth), "--samples", ",".join(str(s) for s in a.samples)]
    res = {"setup": {k: v for k, v in vars(a).items() if k not in ("worker", "variant")},
           "scene": "Sionna RT built-in 'munich'", "backends": []}
    npz = os.path.join(os.path.dirname(a.out) or ".", "sionna_gain_3d.npz")
    for variant in ["llvm_ad_mono_polarized", "cuda_ad_mono_polarized"]:
        cmd = base + ["--variant", variant, "--save-npz", npz]   # same seed -> same map
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = [l for l in p.stdout.splitlines() if l.startswith("RESULT_JSON ")]
        print(p.stdout.replace("RESULT_JSON", "").strip()[-600:])
        if line:
            res["backends"].append(json.loads(line[0][len("RESULT_JSON "):]))
        else:
            print(f"[skip] {variant} failed:\n{p.stderr[-1500:]}")
    by = {b["variant"]: {r["samples_per_tx"]: r["seconds"] for r in b["rows"]} for b in res["backends"]}
    if "cuda_ad_mono_polarized" in by and "llvm_ad_mono_polarized" in by:
        res["speedup"] = {str(s): by["llvm_ad_mono_polarized"][s] / by["cuda_ad_mono_polarized"][s]
                          for s in by["cuda_ad_mono_polarized"] if s in by["llvm_ad_mono_polarized"]}
        print("GPU speed-up per samples/tx:", res["speedup"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()

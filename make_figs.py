"""Poster figures from results/*.json  ->  figs/*.png (300 dpi) and figs/*.pdf (vector).

  python make_figs.py --results results --out figs [--budget-ms 100]

Colour rule: grey = CPU baseline, blue = GPU (ours), aqua = GPU + mixed precision / best.
Every bar carries its value as a direct label (print-safe, colour is never the only cue).
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

GREY, BLUE, AQUA, ORANGE = "#8a8985", "#2a78d6", "#1baf7a", "#eb6834"
INK, INK2 = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13,
    "ytick.labelsize": 13, "legend.fontsize": 13, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": "#e6e5e1",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=300)
    plt.close(fig)
    print(f"[fig] {out}/{name}.png")


def human_time(s):
    return f"{s:.0f} s" if s < 120 else (f"{s / 60:.1f} min" if s < 7200 else f"{s / 3600:.1f} h")


def bar_labels(ax, bars, texts):
    for b, t in zip(bars, texts):
        ax.annotate(t, (b.get_width(), b.get_y() + b.get_height() / 2), xytext=(6, 0),
                    textcoords="offset points", va="center", ha="left", color=INK, fontsize=13)


# ------------------------------------------------------------------------- MARL
def marl(path, out):
    d = json.load(open(path))
    gpu = "cuda" if d["hardware"]["gpus"] else "cpu"
    gname = d["hardware"]["gpus"][0]["name"].replace("NVIDIA ", "") if d["hardware"]["gpus"] else "CPU"

    # (a) throughput vs number of parallel swarms
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for dev, col, lab in (("cpu", GREY, "Vectorised env, CPU"), ("cuda", BLUE, f"Vectorised env, {gname}")):
        rows = sorted((r for r in d["rollout"] if r.get("device") == dev), key=lambda r: r["E"])
        if rows:
            E = [r["E"] for r in rows]; y = [r["env_steps_per_s"] for r in rows]
            ax.plot(E, y, "-o", color=col, lw=2, ms=8, label=lab)
            ax.annotate(f"{y[-1]:,.0f}/s", (E[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", color=INK, fontsize=12)
    conv = [r for r in d["rollout"] if r["impl"] == "numpy_single"]
    if conv:
        c = max(conv, key=lambda r: r["env_steps_per_s"])
        ax.axhline(c["env_steps_per_s"], ls="--", color=INK2, lw=1.5)
        ax.text(1, c["env_steps_per_s"] * 1.25, f"Conventional Gym-style env: {c['env_steps_per_s']:,.0f}/s",
                color=INK2, fontsize=12)
    ax.set(xscale="log", yscale="log", xlabel="Parallel swarms E (4 UAVs each)",
           ylabel="Environment steps / s", title="MARL environment throughput")
    ax.legend(loc="lower right", frameon=False)
    save(fig, out, "marl_throughput")

    # (b) projected wall-clock to the paper's training budget
    p = d.get("projection")
    if p:
        fig, ax = plt.subplots(figsize=(7.5, 3.2))
        labels = ["Conventional\n(CPU env + policy)", f"Vectorised, CPU\n(E={p['best_cpu_vec_E']})",
                  f"Vectorised, GPU\n(E={p['gpu_vec_E']})"]
        vals = [p["conventional_s"], p["best_cpu_vec_s"], p["gpu_vec_s"]]
        bars = ax.barh(labels[::-1], np.array(vals[::-1]) / 60, color=[BLUE, GREY, GREY], height=0.6)
        bar_labels(ax, bars, [human_time(v) + (f"  ({vals[0] / v:.0f}x)" if i == 2 else "")
                              for i, v in enumerate(vals)][::-1])
        ax.set(xlabel="Projected wall-clock (min)",
               title=f"Time for {p['total_env_steps']:.0e} env steps (rollout + PPO update)")
        ax.set_xlim(0, max(vals) / 60 * 1.35); ax.grid(axis="y", visible=False)
        save(fig, out, "marl_projection")

    # (c) scaling with swarm size N
    if d.get("n_sweep"):
        rows = d["n_sweep"]
        N = [r["N"] for r in rows]
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        ax.plot(N, [r["numpy_agent_sps"] for r in rows], "--s", color=GREY, lw=2, ms=8, label="Conventional")
        ax.plot(N, [r["vec_agent_sps"] for r in rows], "-o", color=BLUE, lw=2, ms=8,
                label=f"Vectorised ({gname if gpu == 'cuda' else 'CPU'})")
        for r in rows:
            ax.annotate(f"{r['vec_agent_sps'] / r['numpy_agent_sps']:.0f}x", (r["N"], r["vec_agent_sps"]),
                        xytext=(0, 9), textcoords="offset points", ha="center", color=INK, fontsize=12)
        ax.set(yscale="log", xlabel="UAVs per swarm N", ylabel="UAV-steps / s",
               title="Scaling to larger swarms")
        ax.legend(frameon=False, loc="center right")
        save(fig, out, "marl_nsweep")


# ------------------------------------------------------------------- NSADM / DRM
MODE_LABEL = {"cpu_fp32": "CPU FP32", "gpu_fp32": "GPU FP32\n(CUDA cores)", "gpu_tf32": "GPU TF32\n(Tensor Cores)",
              "gpu_bf16": "GPU BF16 AMP", "gpu_fp16": "GPU FP16 AMP", "gpu_bf16_compile": "BF16 AMP\n+ compile"}


def model_fig(path, out, budget_ms):
    d = json.load(open(path))
    modes = d["modes"]
    title = {"nsadm": "NSADM diffusion (environment reconstruction)",
             "drm": "3D-DRM transformer (radio-map forecasting)"}.get(d["name"], d["name"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    labs = [MODE_LABEL.get(m["mode"], m["mode"]) for m in modes]
    cols = [GREY if m["mode"].startswith("cpu") else BLUE for m in modes]
    best = int(np.argmin([m["train_step"]["median_ms"] for m in modes]))
    cols[best] = AQUA if not modes[best]["mode"].startswith("cpu") else cols[best]
    for ax, key, xl in ((axes[0], "train_step", "Training step (ms, log)"),
                        (axes[1], "infer_latency_b1", "Inference latency, batch 1 (ms, log)")):
        v = [m[key]["median_ms"] for m in modes]
        bars = ax.barh(labs[::-1], v[::-1], color=cols[::-1], height=0.6)
        sp = "train_speedup_vs_cpu" if key == "train_step" else "infer_speedup_vs_cpu"
        bar_labels(ax, bars, [f"{x:,.1f} ms" + (f"  ({m[sp]:.0f}x)" if m.get(sp) and m[sp] > 1.5 else "")
                              for x, m in zip(v, modes)][::-1])
        ax.set_xscale("log"); ax.set_xlim(min(v) / 2, max(v) * 12); ax.set_xlabel(xl)
        ax.grid(axis="y", visible=False)
    if budget_ms:
        axes[1].axvline(budget_ms, color=ORANGE, ls="--", lw=2)
        axes[1].text(budget_ms * 1.08, len(modes) - 0.6, f"real-time budget\n{budget_ms:g} ms",
                     color=INK, fontsize=12, va="top")
    axes[1].set_yticklabels([])
    src = "" if d["source"] == "proxy" else ""
    fig.suptitle(title + src, x=0.02, ha="left", fontsize=16, color=INK)
    fig.tight_layout()
    save(fig, out, f"model_{d['name']}")


def ddp_fig(path, out):
    d = json.load(open(path))
    runs = d["runs"]
    if len(runs) < 2:
        return
    g = [r["gpus"] for r in runs]; y = [r["samples_per_s"] for r in runs]
    fig, ax = plt.subplots(figsize=(6.5, 4.4))
    ax.plot(g, [y[0] * k for k in g], ":", color=INK2, lw=1.5, label="Ideal (linear)")
    ax.plot(g, y, "-o", color=BLUE, lw=2, ms=9, label="Measured (DDP, NCCL)")
    for r in runs:
        ax.annotate(f"{r.get('efficiency', 1) * 100:.0f}%", (r["gpus"], r["samples_per_s"]),
                    xytext=(0, -20), textcoords="offset points", ha="center", color=INK, fontsize=12)
    ax.set(xticks=g, xlabel="GPUs", ylabel="Training samples / s",
           title=f"Multi-GPU scaling: {d['name'].upper()}")
    ax.legend(frameon=False, loc="upper left")
    save(fig, out, f"ddp_{d['name']}")


# ------------------------------------------------------------------- Sionna RT
def sionna_figs(path, npz, out):
    if os.path.exists(path):
        d = json.load(open(path))
        by = {b["variant"]: b["rows"] for b in d["backends"]}
        if by:
            fig, ax = plt.subplots(figsize=(7, 3.6))
            spts = sorted({r["samples_per_tx"] for rows in by.values() for r in rows})
            yy = np.arange(len(spts)); h = 0.38
            for i, (var, col, lab) in enumerate((("llvm_ad_mono_polarized", GREY, "CPU (Dr.Jit LLVM)"),
                                                 ("cuda_ad_mono_polarized", BLUE, "GPU (Dr.Jit CUDA)"))):
                if var not in by:
                    continue
                t = {r["samples_per_tx"]: r["seconds"] for r in by[var]}
                v = [t.get(s, np.nan) for s in spts]
                bars = ax.barh(yy + (h / 2 if i == 0 else -h / 2), v, height=h, color=col, label=lab)
                bar_labels(ax, bars, [human_time(x) for x in v])
            planes = by[next(iter(by))][0]["planes"]
            ax.set_yticks(yy, [f"{s:.0e} rays/GBS" for s in spts])
            ax.set(xlabel="Seconds for full 3D map", xscale="log",
                   title=f"Ray-traced 3D radio map: {planes} altitude planes x 4 GBS")
            ax.set_xlim(right=ax.get_xlim()[1] * 4); ax.grid(axis="y", visible=False)
            ax.legend(frameon=False, loc="lower right")
            save(fig, out, "sionna_time")
    if npz and os.path.exists(npz):
        z = np.load(npz)
        gain, hs, size = z["gain"], z["heights"], float(z["size"])
        # dynamic radio map, Eq. (7): P(p,t) = sum_m g_m(p) P_m(t), sinusoidal BS loads
        t = np.arange(3)
        Pm = 10 ** ((40 + 5 * np.sin(2 * np.pi * (t[:, None] / 6 + np.arange(gain.shape[0])[None] / 4))) / 10)
        pick = [0, len(hs) // 2, len(hs) - 1]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), constrained_layout=True)
        P = np.einsum("mhxy,m->hxy", gain, Pm[0] * 1e-3)               # watts
        PdB = 10 * np.log10(np.maximum(P, 1e-20)) + 30
        vmin, vmax = np.percentile(PdB[PdB > -150], [5, 99.5])
        for ax, k in zip(axes, pick):
            im = ax.imshow(PdB[k].T, origin="lower", cmap="Blues", vmin=vmin, vmax=vmax,
                           extent=[-size / 2, size / 2, -size / 2, size / 2])
            g = z["gbs"]; ax.plot(g[:, 0], g[:, 1], "^", color=ORANGE, ms=10, mec="white")
            ax.set_title(f"altitude {hs[k]:.0f} m"); ax.grid(False)
            ax.set_xlabel("x (m)")
        axes[0].set_ylabel("y (m)")
        cb = fig.colorbar(im, ax=axes, shrink=0.85); cb.set_label("Received power (dBm)")
        fig.suptitle("Ray-traced 3D radio map, Munich (Sionna RT)", x=0.02, ha="left", fontsize=16)
        save(fig, out, "sionna_map")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--budget-ms", type=float, default=None, help="real-time budget line on inference plots")
    a = ap.parse_args()
    R = a.results
    if os.path.exists(f"{R}/marl.json"):
        marl(f"{R}/marl.json", a.out)
    for name in ("nsadm", "drm"):
        if os.path.exists(f"{R}/{name}.json"):
            model_fig(f"{R}/{name}.json", a.out, a.budget_ms)
    for p in glob.glob(f"{R}/ddp_*.json"):
        ddp_fig(p, a.out)
    sionna_figs(f"{R}/sionna.json", f"{R}/sionna_gain_3d.npz", a.out)


if __name__ == "__main__":
    main()

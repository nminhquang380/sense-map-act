"""Training / inference benchmark for the NSADM diffusion denoiser and the 3D-DRM transformer.

Modes (same model, same batch, same weights):
  cpu_fp32      all CPU cores, FP32                         (baseline)
  gpu_fp32      CUDA cores only (TF32 disabled)
  gpu_tf32      Tensor Cores, TF32 matmul/conv
  gpu_bf16      Tensor Cores, automatic mixed precision (bfloat16)
  gpu_fp16      Tensor Cores, automatic mixed precision (float16 + GradScaler)
  gpu_bf16_compile  as gpu_bf16 + torch.compile (kernel fusion)   [--compile]

Per mode we report: train step time, single-request inference latency (batch 1;
for NSADM this is the FULL T-step reverse process), batch inference throughput,
peak GPU memory, and the relative output error versus cpu_fp32 (accuracy check).
Projected full-training time = epochs * ceil(dataset / batch) * step time.

USE YOUR REAL MODEL (recommended for the poster): write a plugin file that defines
    def build(device):
        model = ...                                   # your nn.Module, weights loaded
        def make_batch(bs, device): ...               # returns a batch (any object)
        def train_loss(model, batch): ...             # returns scalar loss
        def infer(model, batch): ...                  # returns output tensor (full pipeline)
        return model, make_batch, train_loss, infer
and run:  python bench_model.py --plugin my_nsadm_plugin.py --name nsadm ...
Without --plugin, size-matched PROXY models are used (good for a sanity check only;
state on the poster which one you measured).
"""
import argparse
import importlib.util
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import fmt_duration, hardware_info, save_json, set_cpu_threads, sync, tf32, time_fn


# ===================================================================== proxy: NSADM
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, c_in), nn.Conv2d(c_in, c_out, 3, padding=1)
        self.t = nn.Linear(t_dim, c_out)
        self.n2, self.c2 = nn.GroupNorm(8, c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x))) + self.t(temb)[:, :, None, None]
        return self.skip(x) + self.c2(F.silu(self.n2(h)))


class ProxyNSADM(nn.Module):
    """Conditional U-Net denoiser R_w(Omega_t, t, C): input = noisy distance matrix +
    conditioning maps Psi (detection prob.) and Sigma_d (CRB variance), Eq. (14)."""

    def __init__(self, ch=64, mults=(1, 2, 4)):
        super().__init__()
        t_dim = ch * 4
        self.temb = nn.Sequential(nn.Linear(ch, t_dim), nn.SiLU(), nn.Linear(t_dim, t_dim))
        self.ch = ch
        self.inp = nn.Conv2d(3, ch, 3, padding=1)
        self.down, self.up = nn.ModuleList(), nn.ModuleList()
        chs, c = [ch], ch
        for m in mults:
            self.down.append(ResBlock(c, ch * m, t_dim)); c = ch * m; chs.append(c)
        self.mid = ResBlock(c, c, t_dim)
        for m in reversed(mults):
            self.up.append(ResBlock(c + chs.pop(), ch * m, t_dim)); c = ch * m
        self.out = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, 1, 3, padding=1))

    def forward(self, x, cond, sigma):
        half = self.ch // 2
        f = torch.exp(-math.log(1e4) * torch.arange(half, device=x.device) / half)
        e = torch.log(sigma)[:, None] * f[None]
        temb = self.temb(torch.cat([e.sin(), e.cos()], -1))
        h = self.inp(torch.cat([x, cond], 1))
        hs = [h]
        for i, blk in enumerate(self.down):
            h = blk(h, temb); hs.append(h)
            if i < len(self.down) - 1:
                h = F.avg_pool2d(h, 2)
        h = self.mid(h, temb)
        for i, blk in enumerate(self.up):
            skip = hs.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = blk(torch.cat([h, skip], 1), temb)
        return self.out(h)


def build_nsadm(device, W=64, H=64, T=50):
    model = ProxyNSADM().to(device)

    def make_batch(bs, device):
        return {"x0": torch.randn(bs, 1, W, H, device=device),
                "cond": torch.rand(bs, 2, W, H, device=device)}

    def train_loss(model, b):
        sigma = torch.exp(torch.randn(b["x0"].shape[0], device=b["x0"].device) * 1.2 - 1.2)
        noisy = b["x0"] + sigma[:, None, None, None] * torch.randn_like(b["x0"])
        return F.mse_loss(model(noisy, b["cond"], sigma), b["x0"])

    @torch.no_grad()
    def infer(model, b):
        # Karras/EDM-style deterministic reverse process with T denoiser calls (Sec. III-B)
        sig = torch.exp(torch.linspace(math.log(80.0), math.log(2e-3), T, device=b["cond"].device))
        g = torch.Generator().manual_seed(0)              # CPU stream -> same noise on every device
        x = torch.randn(b["x0"].shape, generator=g).to(b["cond"].device) * sig[0]
        for i in range(T):
            s = sig[i].expand(x.shape[0])
            x0 = model(x, b["cond"], s)
            s_next = sig[i + 1] if i + 1 < T else torch.zeros((), device=x.device)
            x = x0 + (s_next / sig[i]) * (x - x0)
        return x

    return model, make_batch, train_loss, infer


# =================================================================== proxy: 3D-DRM
class ProxyDRM(nn.Module):
    """Measurement encoder (L_e blocks over UAV measurement tokens, Eqs. (9)-(10)) +
    patch-query decoder with cross-attention (Eqs. (11)-(13)), autoregressive over k frames."""

    def __init__(self, n_patches, patch_vox, d=256, heads=8, le=6, ld=4):
        super().__init__()
        self.meas = nn.Linear(1 + 3 + 1, d)                      # power + 3D position + time
        self.patch = nn.Linear(patch_vox, d)
        self.pos = nn.Parameter(torch.randn(n_patches, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True)
        dec = nn.TransformerDecoderLayer(d, heads, 4 * d, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, le, enable_nested_tensor=False)
        self.dec = nn.TransformerDecoder(dec, ld)
        self.head = nn.Linear(d, patch_vox)

    def encode(self, meas):
        return self.enc(self.meas(meas))

    def decode(self, prev_patches, mem):
        return self.head(self.dec(self.patch(prev_patches) + self.pos, mem))

    def forward(self, meas, maps_in):
        """Teacher-forced training pass (Alg. 1): maps_in (B, k, R, P^3) -> predictions."""
        mem = self.encode(meas)
        return torch.stack([self.decode(maps_in[:, i], mem) for i in range(maps_in.shape[1])], 1)


def build_drm(device, grid=(250, 250, 125), P=10, n_meas=60 * 15, k=5):
    """Defaults follow the ICC paper: 500x500x250 m at 2 m voxels, up to 60 UAVs x 15
    history frames, predict k=5 frames. P = patch side in voxels (set to your value)."""
    gx, gy, gz = (g // P for g in grid)
    n_patches, pv = gx * gy * gz, P ** 3
    model = ProxyDRM(n_patches, pv).to(device)

    def make_batch(bs, device):
        return {"meas": torch.randn(bs, n_meas, 5, device=device),
                "maps": torch.randn(bs, k + 1, n_patches, pv, device=device)}

    def train_loss(model, b):                                   # goes through forward() -> DDP-safe
        return F.mse_loss(model(b["meas"], b["maps"][:, :-1]), b["maps"][:, 1:])

    @torch.no_grad()
    def infer(model, b):
        mem = model.encode(b["meas"])
        cur, outs = b["maps"][:, 0], []
        for _ in range(k):                                      # autoregressive inference
            cur = model.decode(cur, mem)
            outs.append(cur)
        return torch.stack(outs, 1)

    return model, make_batch, train_loss, infer


PROXIES = {"nsadm": build_nsadm, "drm": build_drm}


def load_builder(args):
    if args.plugin:
        spec = importlib.util.spec_from_file_location("plugin", args.plugin)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build, "plugin:" + args.plugin
    return PROXIES[args.name], "proxy"


# ===================================================================== benchmark
def to_dev(b, dev):
    if torch.is_tensor(b):
        return b.to(dev)
    if isinstance(b, dict):
        return {k: to_dev(v, dev) for k, v in b.items()}
    if isinstance(b, (list, tuple)):
        return type(b)(to_dev(v, dev) for v in b)
    return b


def run_mode(mode, builder, args, state_dict, ref_out):
    dev = torch.device("cpu" if mode.startswith("cpu") else "cuda")
    torch.manual_seed(0)
    model, make_batch, train_loss, infer = builder(dev)
    model.load_state_dict(state_dict)
    amp_dtype = {"gpu_bf16": torch.bfloat16, "gpu_fp16": torch.float16,
                 "gpu_bf16_compile": torch.bfloat16}.get(mode)
    use_tf32 = mode != "gpu_fp32"
    base_model = model
    if mode == "gpu_bf16_compile":
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(mode == "gpu_fp16"))
    is_cpu = dev.type == "cpu"
    warm, rep = (1, args.cpu_repeat) if is_cpu else (args.warmup, args.repeat)
    torch.manual_seed(1)
    batch = to_dev(make_batch(args.batch, torch.device("cpu")), dev)
    torch.manual_seed(2)
    b1 = to_dev(make_batch(1, torch.device("cpu")), dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def ctx():
        return torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else torch.autocast("cpu", enabled=False)

    def train_step():
        model.train()
        with ctx():
            loss = train_loss(model, batch)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    def infer1():
        model.eval()
        with ctx():
            return infer(model, b1)

    with tf32(use_tf32):
        r = {"mode": mode}
        r["train_step"] = time_fn(train_step, dev, warm, rep)
        base_model.load_state_dict(state_dict)
        r["infer_latency_b1"] = time_fn(infer1, dev, warm, rep)
        bi = make_batch(args.infer_batch, dev)

        def inferB():
            model.eval()
            with ctx():
                infer(model, bi)
        t = time_fn(inferB, dev, warm, max(2, rep // 2))
        r["infer_throughput_per_s"] = args.infer_batch / (t["median_ms"] / 1e3)
        # accuracy check against the CPU FP32 reference (same weights, same input)
        base_model.load_state_dict(state_dict)
        out = infer1().float().cpu()
        if ref_out is not None:
            r["rel_err_vs_cpu_fp32"] = float((out - ref_out).norm() / (ref_out.norm() + 1e-12))
    if dev.type == "cuda":
        r["peak_mem_GB"] = torch.cuda.max_memory_allocated() / 1e9
    steps_per_epoch = math.ceil(args.dataset / args.batch)
    r["projected_train_s"] = args.epochs * steps_per_epoch * r["train_step"]["median_ms"] / 1e3
    return r, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=["nsadm", "drm"], required=True)
    ap.add_argument("--plugin", default=None, help="file defining build(device); see docstring")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--infer-batch", type=int, default=16)
    ap.add_argument("--dataset", type=int, default=800, help="training samples (NSADM: 1000 scenes; DRM: 120x150 frames)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--modes", default="cpu_fp32,gpu_fp32,gpu_tf32,gpu_bf16,gpu_fp16")
    ap.add_argument("--compile", action="store_true", help="also run gpu_bf16_compile")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--cpu-repeat", type=int, default=3)
    ap.add_argument("--small", action="store_true", help="tiny shapes for a quick smoke test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    set_cpu_threads()
    builder, source = load_builder(args)
    if args.small and source == "proxy":
        base = builder
        builder = (lambda d: base(d, W=32, H=32, T=8)) if args.name == "nsadm" else \
                  (lambda d: base(d, grid=(40, 40, 20), P=10, n_meas=120, k=2))
    modes = args.modes.split(",") + (["gpu_bf16_compile"] if args.compile else [])
    if not torch.cuda.is_available():
        modes = [m for m in modes if m.startswith("cpu")]
        print("[warn] no CUDA device: running CPU modes only")
    torch.manual_seed(0)
    m0 = builder(torch.device("cpu"))[0]
    state = {k: v.clone() for k, v in m0.state_dict().items()}
    n_params = sum(p.numel() for p in m0.parameters())
    del m0

    res = {"name": args.name, "source": source, "params_M": n_params / 1e6, "args": vars(args),
           "hardware": hardware_info(), "modes": []}
    ref = None
    for mode in modes:
        print(f"--- {mode}")
        r, out = run_mode(mode, builder, args, state, ref)
        if mode == "cpu_fp32":
            ref = out
        res["modes"].append(r)
        print(f"  train step {r['train_step']['median_ms']:9.1f} ms | infer b1 "
              f"{r['infer_latency_b1']['median_ms']:9.1f} ms | thr {r['infer_throughput_per_s']:8.1f}/s | "
              f"proj. train {fmt_duration(r['projected_train_s'])}"
              + (f" | rel.err {r['rel_err_vs_cpu_fp32']:.2e}" if 'rel_err_vs_cpu_fp32' in r else ""))
    base = next((m for m in res["modes"] if m["mode"] == "cpu_fp32"), None)
    if base:
        for m in res["modes"]:
            m["train_speedup_vs_cpu"] = base["train_step"]["median_ms"] / m["train_step"]["median_ms"]
            m["infer_speedup_vs_cpu"] = base["infer_latency_b1"]["median_ms"] / m["infer_latency_b1"]["median_ms"]
    save_json(res, args.out or f"results/{args.name}.json")


if __name__ == "__main__":
    main()

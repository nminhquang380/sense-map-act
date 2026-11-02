"""Multi-GPU data-parallel scaling (DistributedDataParallel over NCCL).

Launch once per GPU count, e.g. on the 4x RTX 3090 workstation:
  torchrun --standalone --nproc_per_node=1 bench_ddp.py --name drm --batch 4
  torchrun --standalone --nproc_per_node=2 bench_ddp.py --name drm --batch 4
  torchrun --standalone --nproc_per_node=4 bench_ddp.py --name drm --batch 4
Each run appends to results/ddp_<name>.json. Weak scaling: per-GPU batch fixed, so
the ideal throughput is linear in the number of GPUs; efficiency = measured / ideal.
NCCL all-reduces the gradients after every backward pass (over PCIe or NVLink).
"""
import argparse
import json
import os

import torch
import torch.distributed as dist

from bench_model import PROXIES, load_builder, to_dev
from common import hardware_info, time_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", choices=list(PROXIES), required=True)
    ap.add_argument("--plugin", default=None)
    ap.add_argument("--batch", type=int, default=4, help="per-GPU batch")
    ap.add_argument("--amp", default="bf16", choices=["none", "bf16"])
    ap.add_argument("--repeat", type=int, default=20)
    ap.add_argument("--out", default=None)
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()

    cuda = torch.cuda.is_available()
    dist.init_process_group("nccl" if cuda else "gloo")      # gloo/CPU only for smoke tests
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", 0))
    if cuda:
        torch.cuda.set_device(local)
    dev = torch.device("cuda", local) if cuda else torch.device("cpu")
    builder, source = load_builder(args)
    if args.small and source == "proxy":
        base = builder
        builder = (lambda d: base(d, W=32, H=32, T=8)) if args.name == "nsadm" else \
                  (lambda d: base(d, grid=(40, 40, 20), P=10, n_meas=120, k=2))
    torch.manual_seed(0)
    model, make_batch, train_loss, _ = builder(dev)
    ddp = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local] if cuda else None)
    opt = torch.optim.AdamW(ddp.parameters(), lr=1e-4)
    torch.manual_seed(1 + rank)
    batch = to_dev(make_batch(args.batch, torch.device("cpu")), dev)
    amp = torch.bfloat16 if (args.amp == "bf16" and cuda) else None

    def step():
        with torch.autocast(dev.type, dtype=amp or torch.bfloat16, enabled=amp is not None):
            loss = train_loss(ddp, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    t = time_fn(step, dev, warmup=5, repeat=args.repeat)
    # slowest rank defines the step time
    ms = torch.tensor([t["median_ms"]], device=dev)
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    if rank == 0:
        out = args.out or f"results/ddp_{args.name}.json"
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        data = json.load(open(out)) if os.path.exists(out) else {"name": args.name, "source": source,
                                                                 "hardware": hardware_info(), "runs": []}
        data["runs"] = [r for r in data["runs"] if r["gpus"] != world]
        data["runs"].append({"gpus": world, "per_gpu_batch": args.batch, "amp": args.amp,
                             "step_ms": ms.item(),
                             "samples_per_s": world * args.batch / (ms.item() / 1e3)})
        data["runs"].sort(key=lambda r: r["gpus"])
        one = next((r for r in data["runs"] if r["gpus"] == 1), None)
        for r in data["runs"]:
            if one:
                r["speedup"] = r["samples_per_s"] / one["samples_per_s"]
                r["efficiency"] = r["speedup"] / r["gpus"]
        json.dump(data, open(out, "w"), indent=2)
        print(f"[{world} GPU] step {ms.item():.1f} ms, {world * args.batch / (ms.item() / 1e3):.1f} samples/s -> {out}")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

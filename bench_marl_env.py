"""GPU-vectorised multi-cell LAWN congestion environment + MAPPO rollout benchmark.

Re-implements the environment of
  N. D. M. Quang et al., "Game-theoretic MARL for swarm trajectory planning in
  low-altitude wireless networks," IEEE ICCC 2026 (arXiv:2606.16386),
twice:
  (1) NumpyCongestionEnv  - conventional Gym-style env: one episode, Python loop
                            over UAVs, policy queried through CPU<->GPU copies;
  (2) VecCongestionEnv    - every quantity is a batched tensor of shape (E, N, ...),
                            so E independent swarms step in ONE set of kernels and the
                            env, the actor and the critic all stay on the GPU.

Both implement the same MDP (Eqs. (1)-(3), (7)-(9), (14)-(18) of the paper):
  nearest-GBS association, equal RB split floor(K_b / n_b), LoS rate, EMA rate,
  product utility U_i = R_hat_i * T_min_i / T_i, per-step reward
  r = r_rate + r_nav + r_col + r_term, per-UAV reset on arrival / timeout.
`--check` verifies that, fed the same missions and actions, both envs return the same
rewards (float64). Because the MDP is identical, the learning curve versus
environment steps is unchanged; only wall-clock per step changes. This is what lets
us project end-to-end training time from measured throughput.

Physical constants marked PLACEHOLDER are not given in the paper's Table I; they do
not affect throughput. Replace them with your simulator's values if you also want
the rates to match the paper.

Usage
  python bench_marl_env.py --check                 # equivalence test (CPU is fine)
  python bench_marl_env.py --out results/marl.json # full benchmark
"""
import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn

from common import fmt_duration, hardware_info, save_json, set_cpu_threads, sync

# ----------------------------------------------------------------------------- config
CFG = dict(
    N=4,                 # UAVs per swarm (Table I)
    M=4,                 # GBSs (Table I)
    W=100, L=100, H=30,  # voxel grid (Table I)
    ds=10.0,             # voxel size [m] (Table I)
    vmax=3,              # max voxel steps per slot, ||a||_1 <= vmax (Table I)
    Tmax=100,            # episode / mission deadline [slots] (Table I)
    K=10,                # RBs per GBS (Table I)
    df=180e3,            # RB bandwidth [Hz] (Table I)
    alpha=2.0,           # path-loss exponent (Table I)
    gbs_height=25.0,     # PLACEHOLDER [m]
    ptx_dbm=23.0,        # PLACEHOLDER per-RB transmit power
    beta0_db=-38.5,      # PLACEHOLDER reference gain at 1 m (~FSPL @ 2 GHz)
    noise_dbm=-174 + 10 * math.log10(180e3) + 7.0,  # PLACEHOLDER per-RB noise (NF 7 dB)
    beta_ema=0.1,        # EMA smoothing factor beta, Eq. (7) (PLACEHOLDER value)
    omega=0.1,           # navigation shaping weight, Eq. (16) (PLACEHOLDER value)
    rho_col=1.0,         # collision penalty, Eq. (17) (PLACEHOLDER value)
    r_fail=1.0,          # timeout penalty, Eq. (18) (PLACEHOLDER value)
)


def action_table(vmax):
    """All integer displacements a in Z^3 with ||a||_1 <= vmax (63 actions for vmax=3)."""
    acts = [(x, y, z)
            for x in range(-vmax, vmax + 1)
            for y in range(-vmax, vmax + 1)
            for z in range(-vmax, vmax + 1)
            if abs(x) + abs(y) + abs(z) <= vmax]
    return np.array(acts, dtype=np.int64)


def gbs_positions(cfg):
    """M GBSs on a regular ground grid (voxel units for x, y; metres for height)."""
    m = int(round(math.sqrt(cfg["M"])))
    xs = [(i + 0.5) * cfg["W"] / m for i in range(m)]
    ys = [(j + 0.5) * cfg["L"] / m for j in range(m)]
    pts = [(x, y) for x in xs for y in ys][: cfg["M"]]
    return np.array(pts, dtype=np.float64)


def snr_ref(cfg):
    """P_tx * beta0 / sigma^2 (linear)."""
    return 10 ** ((cfg["ptx_dbm"] + cfg["beta0_db"] - cfg["noise_dbm"]) / 10)


class MissionBank:
    """Pre-drawn (origin, destination) pairs, consumed in order. Used by --check so
    both env implementations see exactly the same missions."""

    def __init__(self, n, cfg, seed=0):
        rng = np.random.default_rng(seed)
        dims = np.array([cfg["W"], cfg["L"], cfg["H"]])
        o = rng.integers(0, dims, size=(n, 3))
        d = rng.integers(0, dims, size=(n, 3))
        same = (o == d).all(1)
        d[same, 0] = (d[same, 0] + 1) % cfg["W"]
        self.o, self.d, self.ptr = o, d, 0

    def take(self, k):
        s = slice(self.ptr, self.ptr + k)
        self.ptr += k
        return self.o[s], self.d[s]


# ------------------------------------------------------------- (1) conventional env
class NumpyCongestionEnv:
    """Gym-style single environment, per-UAV Python loop (the usual research code)."""

    def __init__(self, cfg, seed=0, bank=None):
        self.c = cfg
        self.N, self.M = cfg["N"], cfg["M"]
        self.dims = np.array([cfg["W"], cfg["L"], cfg["H"]])
        self.A = action_table(cfg["vmax"])
        self.g = gbs_positions(cfg)
        self.snr0 = snr_ref(cfg)
        self.rng = np.random.default_rng(seed)
        self.bank = bank
        self.reset()

    def _new_mission(self):
        if self.bank is not None:
            o, d = self.bank.take(1)
            return o[0].copy(), d[0].copy()
        o = self.rng.integers(0, self.dims)
        d = self.rng.integers(0, self.dims)
        while (o == d).all():
            d = self.rng.integers(0, self.dims)
        return o, d

    def _start(self, i):
        o, d = self._new_mission()
        self.q[i], self.d[i] = o, d
        self.t[i], self.rbar[i] = 0, 0.0
        self.tmin[i] = math.ceil(np.abs(o - d).sum() / self.c["vmax"])

    def reset(self):
        self.q = np.zeros((self.N, 3), dtype=np.int64)
        self.d = np.zeros((self.N, 3), dtype=np.int64)
        self.t = np.zeros(self.N, dtype=np.int64)
        self.rbar = np.zeros(self.N)
        self.tmin = np.zeros(self.N)
        for i in range(self.N):
            self._start(i)
        return self.obs()

    def _pos_m(self, q):
        return np.array([q[0] * self.c["ds"], q[1] * self.c["ds"], q[2] * self.c["ds"]], dtype=np.float64)

    def obs(self):
        loads = np.zeros(self.M)
        for i in range(self.N):
            loads[self._assoc(self.q[i])[0]] += 1
        o = []
        for i in range(self.N):
            o.append(np.concatenate([
                self.q[i] / self.dims, self.d[i] / self.dims,
                [self.tmin[i] / self.c["Tmax"], self.rbar[i] / 50.0],
                loads / self.N,
                np.concatenate([self.g / self.dims[:2], np.full((self.M, 1), 0.0)], 1).ravel(),
            ]))
        return np.stack(o).astype(np.float32)

    def _assoc(self, q):
        p = self._pos_m(q)
        gm = np.concatenate([self.g * self.c["ds"], np.full((self.M, 1), self.c["gbs_height"])], 1)
        dist = np.sqrt(((gm - p) ** 2).sum(1))
        b = int(np.argmin(dist))
        return b, dist[b]

    def step(self, actions):
        c, N = self.c, self.N
        q_new = np.clip(self.q + self.A[actions], 0, self.dims - 1)
        # association and cell loads at t+1
        assoc = [self._assoc(q_new[i]) for i in range(N)]
        loads = np.zeros(self.M, dtype=np.int64)
        for b, _ in assoc:
            loads[b] += 1
        rewards = np.zeros(N)
        done = np.zeros(N, dtype=bool)
        for i in range(N):
            b, dist = assoc[i]
            kappa = math.floor(c["K"] / loads[b])
            rate = kappa * c["df"] * math.log2(1 + self.snr0 / dist ** c["alpha"]) / 1e6  # Mbps
            rbar_new = (1 - c["beta_ema"]) * self.rbar[i] + c["beta_ema"] * rate
            r = rbar_new - self.rbar[i]
            r += c["omega"] * (np.linalg.norm(self.q[i] - self.d[i]) - np.linalg.norm(q_new[i] - self.d[i]))
            for j in range(N):
                if j != i and (q_new[j] == q_new[i]).all():
                    r -= c["rho_col"]
                    break
            t_i = self.t[i] + 1
            if (q_new[i] == self.d[i]).all():
                rhat = rbar_new / (1 - (1 - c["beta_ema"]) ** t_i)
                r += rhat * self.tmin[i] / t_i
                done[i] = True
            elif t_i >= c["Tmax"]:
                r -= c["r_fail"]
                done[i] = True
            rewards[i] = r
            self.rbar[i], self.t[i] = rbar_new, t_i
        self.q = q_new
        for i in range(N):
            if done[i]:
                self._start(i)
        return self.obs(), rewards, done


# ------------------------------------------------------------- (2) vectorised env
class VecCongestionEnv:
    """E independent swarms as batched tensors; runs on CPU or GPU unchanged."""

    def __init__(self, E, cfg, device="cuda", dtype=torch.float32, seed=0, bank=None):
        self.E, self.c, self.dev, self.dt = E, cfg, torch.device(device), dtype
        self.N, self.M = cfg["N"], cfg["M"]
        self.dims = torch.tensor([cfg["W"], cfg["L"], cfg["H"]], device=self.dev)
        self.A = torch.as_tensor(action_table(cfg["vmax"]), device=self.dev)
        g = gbs_positions(cfg)
        self.g_vox = torch.as_tensor(g, device=self.dev, dtype=dtype)
        self.g_m = torch.cat([self.g_vox * cfg["ds"],
                              torch.full((self.M, 1), cfg["gbs_height"], device=self.dev, dtype=dtype)], 1)
        self.snr0 = snr_ref(cfg)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.bank = bank
        g_feat = torch.cat([self.g_vox / self.dims[:2], torch.zeros(self.M, 1, device=self.dev)], 1).reshape(-1)
        self.g_feat = g_feat.to(torch.float32).expand(E, self.N, -1)
        self.q = torch.zeros(E, self.N, 3, dtype=torch.long, device=self.dev)
        self.d = torch.zeros_like(self.q)
        self.t = torch.zeros(E, self.N, dtype=torch.long, device=self.dev)
        self.rbar = torch.zeros(E, self.N, dtype=dtype, device=self.dev)
        self.tmin = torch.zeros(E, self.N, dtype=dtype, device=self.dev)
        self._start(torch.ones(E, self.N, dtype=torch.bool, device=self.dev))

    def _draw(self, k):
        if self.bank is not None:
            o, d = self.bank.take(k)
            return (torch.as_tensor(o, device=self.dev), torch.as_tensor(d, device=self.dev))
        o = (torch.rand(k, 3, generator=self.gen, device=self.dev) * self.dims).long()
        d = (torch.rand(k, 3, generator=self.gen, device=self.dev) * self.dims).long()
        same = (o == d).all(1)
        d[same, 0] = (d[same, 0] + 1) % self.c["W"]
        return o, d

    def _start(self, mask):
        # row-major nonzero order == (env, uav) order == NumpyCongestionEnv loop order
        k = int(mask.sum()) if self.bank is not None else None
        if self.bank is not None:
            if k == 0:
                return
            o, d = self._draw(k)
            self.q[mask], self.d[mask] = o, d
        else:  # sync-free path: draw for all, select with where (no host round-trip)
            o, d = self._draw(self.E * self.N)
            o, d = o.view(self.E, self.N, 3), d.view(self.E, self.N, 3)
            m3 = mask.unsqueeze(-1)
            self.q = torch.where(m3, o, self.q)
            self.d = torch.where(m3, d, self.d)
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)
        self.rbar = torch.where(mask, torch.zeros_like(self.rbar), self.rbar)
        tmin = torch.ceil((self.q - self.d).abs().sum(-1).to(self.dt) / self.c["vmax"])
        self.tmin = torch.where(mask, tmin, self.tmin)

    def _assoc(self, q):
        p = q.to(self.dt) * self.c["ds"]                                   # (E,N,3) metres
        dist = torch.linalg.norm(p.unsqueeze(2) - self.g_m, dim=-1)       # (E,N,M)
        dmin, b = dist.min(-1)
        return b, dmin

    def obs(self):
        b, _ = self._assoc(self.q)
        loads = torch.nn.functional.one_hot(b, self.M).sum(1).to(torch.float32) / self.N  # (E,M)
        dimsf = self.dims.to(torch.float32)
        return torch.cat([
            self.q.to(torch.float32) / dimsf, self.d.to(torch.float32) / dimsf,
            (self.tmin / self.c["Tmax"]).unsqueeze(-1).to(torch.float32),
            (self.rbar / 50.0).unsqueeze(-1).to(torch.float32),
            loads.unsqueeze(1).expand(-1, self.N, -1), self.g_feat,
        ], -1)

    def step(self, actions):
        c = self.c
        q_new = torch.minimum(torch.clamp(self.q + self.A[actions], min=0), self.dims - 1)
        b, dist = self._assoc(q_new)
        loads = torch.nn.functional.one_hot(b, self.M).sum(1)             # (E,M)
        n_i = torch.gather(loads, 1, b)                                    # (E,N)
        kappa = torch.div(c["K"], n_i, rounding_mode="floor").to(self.dt)  # Eq. (1)
        rate = kappa * c["df"] * torch.log2(1 + self.snr0 / dist ** c["alpha"]) / 1e6  # Eq. (2)
        rbar_new = (1 - c["beta_ema"]) * self.rbar + c["beta_ema"] * rate             # Eq. (7)
        r = rbar_new - self.rbar                                                       # Eq. (15)
        dold = torch.linalg.norm((self.q - self.d).to(self.dt), dim=-1)
        dnew = torch.linalg.norm((q_new - self.d).to(self.dt), dim=-1)
        r = r + c["omega"] * (dold - dnew)                                             # Eq. (16)
        same = (q_new.unsqueeze(2) == q_new.unsqueeze(1)).all(-1)                      # (E,N,N)
        same = same & ~torch.eye(self.N, dtype=torch.bool, device=self.dev)
        r = r - c["rho_col"] * same.any(-1).to(self.dt)                                # Eq. (17)
        t_i = self.t + 1
        arrived = (q_new == self.d).all(-1)
        timeout = (t_i >= c["Tmax"]) & ~arrived
        rhat = rbar_new / (1 - (1 - c["beta_ema"]) ** t_i.to(self.dt))
        util = rhat * self.tmin / t_i.to(self.dt)                                      # Eq. (9)
        r = r + torch.where(arrived, util, torch.zeros_like(util)) - c["r_fail"] * timeout.to(self.dt)  # Eq. (18)
        self.q, self.rbar, self.t = q_new, rbar_new, t_i
        done = arrived | timeout
        self._start(done)
        return self.obs(), r, done


# ------------------------------------------------------------------- MAPPO networks
def mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


def make_nets(cfg, device):
    obs_dim = 8 + 4 * cfg["M"]
    n_act = len(action_table(cfg["vmax"]))
    actor = mlp(obs_dim, n_act).to(device)             # shared decentralised actor (2x256, Sec. IV-B)
    critic = mlp(cfg["N"] * obs_dim, 1).to(device)     # centralised critic on the global state
    return actor, critic, obs_dim, n_act


# ---------------------------------------------------------------------- benchmarks
@torch.no_grad()
def rollout_numpy(cfg, steps, policy_dev):
    """Conventional pipeline: CPU env, policy on `policy_dev`, copies every step."""
    env = NumpyCongestionEnv(cfg, seed=1)
    actor, critic, _, _ = make_nets(cfg, policy_dev)
    o = env.reset()

    def one():
        nonlocal o
        ot = torch.from_numpy(o).to(policy_dev)
        logits = actor(ot)
        critic(ot.reshape(1, -1))
        a = torch.distributions.Categorical(logits=logits).sample().cpu().numpy()
        o, _, _ = env.step(a)

    for _ in range(20):
        one()
    sync(policy_dev)
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    sync(policy_dev)
    return steps / (time.perf_counter() - t0)


@torch.no_grad()
def rollout_vec(cfg, E, steps, device):
    env = VecCongestionEnv(E, cfg, device=device, seed=1)
    actor, critic, _, _ = make_nets(cfg, device)
    o = env.obs()

    def one():
        nonlocal o
        logits = actor(o)
        critic(o.reshape(E, -1))
        a = torch.distributions.Categorical(logits=logits).sample()
        o, _, _ = env.step(a)

    for _ in range(10):
        one()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(steps):
        one()
    sync(device)
    return E * steps / (time.perf_counter() - t0)


def ppo_update_throughput(cfg, device, n_samples=32768, minibatch=4096, epochs=4):
    """Samples*epochs processed per second by the PPO update (actor + critic fwd/bwd)."""
    actor, critic, obs_dim, n_act = make_nets(cfg, device)
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)
    obs = torch.randn(n_samples, obs_dim, device=device)
    gstate = torch.randn(n_samples, cfg["N"] * obs_dim, device=device)
    act = torch.randint(0, n_act, (n_samples,), device=device)
    adv = torch.randn(n_samples, device=device)
    ret = torch.randn(n_samples, device=device)
    old_lp = torch.randn(n_samples, device=device).clamp(max=0)

    def update():
        for _ in range(epochs):
            perm = torch.randperm(n_samples, device=device)
            for s in range(0, n_samples, minibatch):
                idx = perm[s:s + minibatch]
                dist = torch.distributions.Categorical(logits=actor(obs[idx]))
                ratio = torch.exp(dist.log_prob(act[idx]) - old_lp[idx])
                l_clip = torch.min(ratio * adv[idx], ratio.clamp(0.8, 1.2) * adv[idx]).mean()
                l_v = ((critic(gstate[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
                loss = -l_clip + 0.5 * l_v - 0.01 * dist.entropy().mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

    update()
    sync(device)
    t0 = time.perf_counter()
    update()
    sync(device)
    return n_samples * epochs / (time.perf_counter() - t0)


def equivalence_check(cfg, steps=300, seed=0):
    """Same missions + same actions -> same rewards in both implementations (float64)."""
    n_missions = 10000
    env_np = NumpyCongestionEnv(cfg, bank=MissionBank(n_missions, cfg, seed))
    env_t = VecCongestionEnv(1, cfg, device="cpu", dtype=torch.float64, bank=MissionBank(n_missions, cfg, seed))
    rng = np.random.default_rng(seed + 1)
    n_act = len(action_table(cfg["vmax"]))
    A = action_table(cfg["vmax"])
    max_err, n_done = 0.0, 0
    for _ in range(steps):
        a = rng.integers(0, n_act, size=cfg["N"])
        # 70% greedy moves toward the destination, so the arrival / utility branch is exercised
        greedy = np.abs(env_np.q[:, None, :] + A[None] - env_np.d[:, None, :]).sum(-1).argmin(1)
        use = rng.random(cfg["N"]) < 0.7
        a = np.where(use, greedy, a)
        o1, r1, d1 = env_np.step(a)
        o2, r2, d2 = env_t.step(torch.as_tensor(a).unsqueeze(0))
        max_err = max(max_err, float(np.abs(r1 - r2[0].numpy()).max()),
                      float(np.abs(o1 - o2[0].numpy()).max()))
        assert (d1 == d2[0].numpy()).all(), "done flags differ"
        n_done += int(d1.sum())
    return {"steps": steps, "mission_resets": n_done, "max_abs_error": max_err, "passed": max_err < 1e-4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only run the equivalence test")
    ap.add_argument("--out", default="results/marl.json")
    ap.add_argument("--steps", type=int, default=200, help="timed steps per configuration")
    ap.add_argument("--np-steps", type=int, default=2000, help="timed steps for the conventional env")
    ap.add_argument("--envs", default="1,16,64,256,1024,4096,16384")
    ap.add_argument("--cpu-envs", default="1,16,64,256,1024")
    ap.add_argument("--total-steps", type=float, default=2e6, help="env steps for projection (paper Fig. 2)")
    ap.add_argument("--ppo-epochs", type=int, default=4)
    ap.add_argument("--n-sweep", default="4,16,40,64", help="swarm sizes N for the scaling sweep ('' to skip)")
    ap.add_argument("--sweep-uav-budget", type=int, default=262144, help="E*N kept ~constant on GPU in the sweep")
    args = ap.parse_args()

    cfg = dict(CFG)
    chk = equivalence_check(cfg)
    print(f"[equivalence] {chk}")
    if args.check:
        return
    set_cpu_threads()
    has_gpu = torch.cuda.is_available()
    gdev = "cuda" if has_gpu else "cpu"
    res = {"hardware": hardware_info(), "config": cfg, "equivalence": chk, "rollout": [], "ppo_update": {}}

    sps = rollout_numpy(cfg, args.np_steps, "cpu")
    res["rollout"].append({"impl": "numpy_single", "policy": "cpu", "E": 1, "env_steps_per_s": sps})
    print(f"numpy single env, policy CPU : {sps:10.0f} env-steps/s")
    if has_gpu:
        sps = rollout_numpy(cfg, args.np_steps, "cuda")
        res["rollout"].append({"impl": "numpy_single", "policy": "cuda", "E": 1, "env_steps_per_s": sps})
        print(f"numpy single env, policy GPU : {sps:10.0f} env-steps/s")

    for E in map(int, args.cpu_envs.split(",")):
        sps = rollout_vec(cfg, E, max(20, args.steps // 4), "cpu")
        res["rollout"].append({"impl": "torch_vec", "device": "cpu", "E": E, "env_steps_per_s": sps})
        print(f"torch vec CPU  E={E:6d}     : {sps:10.0f} env-steps/s")
    if has_gpu:
        for E in map(int, args.envs.split(",")):
            sps = rollout_vec(cfg, E, args.steps, "cuda")
            res["rollout"].append({"impl": "torch_vec", "device": "cuda", "E": E, "env_steps_per_s": sps})
            print(f"torch vec GPU  E={E:6d}     : {sps:10.0f} env-steps/s")

    for dev in (["cpu", "cuda"] if has_gpu else ["cpu"]):
        res["ppo_update"][dev] = ppo_update_throughput(cfg, dev, epochs=args.ppo_epochs)
        print(f"PPO update {dev:4s}: {res['ppo_update'][dev]:10.0f} sample-epochs/s")

    # projected end-to-end training time = rollout time + update time
    T, N, K = args.total_steps, cfg["N"], args.ppo_epochs
    def proj(sps, dev):
        return T / sps + T * N * K / res["ppo_update"][dev]
    base = [r for r in res["rollout"] if r["impl"] == "numpy_single" and r.get("policy") == gdev][0]
    best_cpu = max((r for r in res["rollout"] if r.get("device") == "cpu"), key=lambda r: r["env_steps_per_s"])
    best_gpu = max((r for r in res["rollout"] if r.get("device") == gdev), key=lambda r: r["env_steps_per_s"])
    res["projection"] = {
        "total_env_steps": T,
        "conventional_s": proj(base["env_steps_per_s"], gdev),
        "best_cpu_vec_s": proj(best_cpu["env_steps_per_s"], "cpu"),
        "best_cpu_vec_E": best_cpu["E"],
        "gpu_vec_s": proj(best_gpu["env_steps_per_s"], gdev),
        "gpu_vec_E": best_gpu["E"],
        "note": "projection = T/rollout_throughput + T*N*epochs/update_throughput; "
                "valid because --check shows the vectorised env is the same MDP.",
    }
    p = res["projection"]
    print(f"\nProjected time for {T:.0e} env steps:")
    print(f"  conventional (numpy env + {gdev} policy): {fmt_duration(p['conventional_s'])}")
    print(f"  best CPU vectorised (E={p['best_cpu_vec_E']}):     {fmt_duration(p['best_cpu_vec_s'])}")
    print(f"  GPU vectorised (E={p['gpu_vec_E']}):          {fmt_duration(p['gpu_vec_s'])}")
    print(f"  speed-up vs conventional: {p['conventional_s'] / p['gpu_vec_s']:.1f}x, "
          f"vs best CPU: {p['best_cpu_vec_s'] / p['gpu_vec_s']:.1f}x")

    # swarm-size sweep: conventional per-UAV loops scale as O(N^2) (collision check);
    # the vectorised env evaluates all pairs of all swarms in one kernel.
    res["n_sweep"] = []
    for N in [int(x) for x in args.n_sweep.split(",") if x]:
        c2 = dict(cfg, N=N)
        row = {"N": N,
               "numpy_single_sps": rollout_numpy(c2, max(100, args.np_steps // max(1, N // 4)), gdev)}
        E = max(1, args.sweep_uav_budget // N) if has_gpu else 64
        row["vec_E"] = E
        row["vec_sps"] = rollout_vec(c2, E, max(20, args.steps // 2), gdev)
        row["vec_device"] = gdev
        # agent-steps/s = env-steps/s * N  (one env step moves all N UAVs)
        row["numpy_agent_sps"] = row["numpy_single_sps"] * N
        row["vec_agent_sps"] = row["vec_sps"] * N
        res["n_sweep"].append(row)
        print(f"N={N:3d}: numpy {row['numpy_agent_sps']:10.0f} UAV-steps/s | "
              f"vec {gdev} (E={E}) {row['vec_agent_sps']:12.0f} UAV-steps/s "
              f"-> {row['vec_agent_sps'] / row['numpy_agent_sps']:.0f}x")
    save_json(res, args.out)


if __name__ == "__main__":
    main()

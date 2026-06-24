"""Gate 1 — whitened-marginal sanity (README §8).

SONAR is not a KL-regularized VAE latent. Before trusting the conditional model,
fit a TINY unconditional flow to the whitened marginal of the target latents,
sample, decode, and check coherence. If this can't reproduce SONAR's own
distribution, the problem is the space/flow, not the predictor.

    python -m Sync.scripts.gate_uncond_flow --task gigaword --smoke
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from ..codec import SonarCodecAdapter
from ..components import TimestepEmbedding
from ..config import get_preset
from ..data import Whitening
from ..metrics import SemanticScorer
from ..predictor import flow_matching_target


class TinyUncondFlow(nn.Module):
    def __init__(self, d: int, hidden: int = 2048, time_dim: int = 256):
        super().__init__()
        self.time = TimestepEmbedding(time_dim, hidden)
        self.in_proj = nn.Linear(d, hidden)
        self.net = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
        )
        self.out = nn.Linear(hidden, d)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(z) + self.time(t)
        return self.out(self.net(h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gigaword")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--sample-steps", type=int, default=50)
    ap.add_argument("--n-samples", type=int, default=200)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps = 200
        args.n_samples = 32

    device = args.device if torch.cuda.is_available() else "cpu"
    cfg = get_preset(args.task)
    cfg.validate()
    cache = cfg.data.latent_cache_dir
    d, q, M = cfg.predictor.latent_dim, cfg.predictor.latents_per_chunk, cfg.predictor.n_tgt_chunks

    meta = json.load(open(os.path.join(cache, "meta.json")))
    num = meta["num"]
    m_arr = np.load(os.path.join(cache, "m.npy"))
    target = np.memmap(
        os.path.join(cache, "target.f32"), dtype="float32", mode="r",
        shape=(num, M, q, d),
    )
    whitening = Whitening.load(os.path.join(cache, "whitening.npz")).to(device)

    # gather + whiten the valid target latents
    pool = []
    for i in range(num):
        mi = int(m_arr[i])
        if mi > 0:
            pool.append(np.ascontiguousarray(target[i, :mi]).reshape(-1, d))
    Z0 = torch.from_numpy(np.concatenate(pool, axis=0)).float().to(device)
    Z0 = whitening.apply(Z0)
    print(f"marginal latents: {Z0.shape[0]} x {d}")

    model = TinyUncondFlow(d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    model.train()
    N = Z0.shape[0]
    for step in range(args.steps):
        idx = torch.randint(0, N, (args.batch,), device=device)
        z0 = Z0[idx]
        t = torch.rand(args.batch, device=device)
        eps = torch.randn_like(z0)
        z_t, v = flow_matching_target(z0[:, None, :], eps[:, None, :], t)
        v_hat = model(z_t[:, 0], t)
        loss = ((v_hat - v[:, 0]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 200 == 0:
            print(f"step {step} | loss {loss.item():.4f}")

    # sample
    model.eval()
    with torch.no_grad():
        Z = torch.randn(args.n_samples, d, device=device)
        dt = 1.0 / args.sample_steps
        for k in range(args.sample_steps):
            t = torch.full((args.n_samples,), k * dt, device=device)
            Z = Z + dt * model(Z, t)
        Z_un = whitening.invert(Z)

    codec = SonarCodecAdapter(cfg.codec, cfg.tokenizer, device=device)
    decoded = codec.decode_latents(Z_un[:, None, :])

    # coherence: each generated sentence's best semantic sim to a pool of real targets
    refs = []
    with open(os.path.join(cache, "refs.jsonl")) as f:
        for line in f:
            refs.extend(json.loads(line)["tgt"])
    refs = refs[:2000]
    scorer = SemanticScorer()
    de = scorer.embed(decoded)
    re = scorer.embed(refs)
    best = (de @ re.T).max(dim=1).values
    print(f"\nGate 1 coherence (mean best sem-sim to real targets): {float(best.mean()):.4f}")
    print("sample decodes:")
    for s in decoded[:10]:
        print("  -", s)


if __name__ == "__main__":
    main()

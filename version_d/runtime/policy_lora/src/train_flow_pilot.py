"""
train_flow_pilot.py — 在离线 cache 上训练 FlowMatchingActionHead（CPU 即可）。

这是 pilot 的"训练"阶段，不碰 base 模型、不碰 LIBERO 仿真，只读 cache 张量。
强基线 = path_type='straight'（其 1 步 Euler 即残差 MLP，与 gaussian 同参数架构）；
创新路径 = path_type='gaussian'（T1 curved/stochastic CFM）。

用法:
  python src/train_flow_pilot.py --cache pilot_cache.pt --path-type straight --out head_straight.pt
  python src/train_flow_pilot.py --cache pilot_cache.pt --path-type gaussian --out head_gaussian.pt
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[0]


def _load_head_module():
    spec = importlib.util.spec_from_file_location(
        "flow_matching_action_head", _SRC / "models/flow_matching_action_head.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flow_matching_action_head"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--path-type", choices=["straight", "gaussian"], default="gaussian")
    ap.add_argument("--sigma", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--cond-dim", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sys.path.insert(0, str(_SRC))
    from pilot_cache import load_cache, split_cache

    FlowMatchingActionHead = _load_head_module().FlowMatchingActionHead

    d = load_cache(args.cache)
    tr, _va = split_cache(d)
    D = d["a0"].shape[-1]
    T = d["a0"].shape[1]

    head = FlowMatchingActionHead(
        action_dim=D,
        horizon=T,
        cond_dim=args.cond_dim,
        path_type=args.path_type,
        sigma=args.sigma,
        num_layers=args.num_layers,
    ).to(args.device)

    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    a0 = tr["a0"].to(args.device)
    c = tr["c"].to(args.device)
    a1 = tr["a_gt"].to(args.device)
    n = a0.shape[0]

    for ep in range(args.epochs):
        perm = torch.randperm(n, device=args.device)
        for s in range(0, n, args.batch_size):
            idx = perm[s : s + args.batch_size]
            loss = head.flow_loss(a0[idx], a1[idx], c[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % 5 == 0 or args.epochs <= 5:
            print(f"  epoch {ep+1}/{args.epochs}  loss={loss.item():.4f}")

    torch.save(
        {
            "state": head.state_dict(),
            "path_type": args.path_type,
            "sigma": args.sigma,
            "action_dim": D,
            "horizon": T,
            "cond_dim": args.cond_dim,
        },
        args.out,
    )
    print(f"[train] saved {args.path_type} head -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

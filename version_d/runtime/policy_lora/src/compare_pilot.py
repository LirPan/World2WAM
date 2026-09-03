"""
compare_pilot.py — pilot 对比：残差 MLP (straight, 1步) vs Flow (gaussian, 1/4/10步)。

这是 pilot 的"判定"阶段，回答学长那句"本质还是 MLP"：
  - straight(1步) 与 gaussian(1步) 同架构、同参，但 straight 的 1 步 Euler 即残差 MLP；
  - 若 gaussian(10步) 的 MSE 明显 < straight(1步)，说明 T1 的 curved 路径让多步集成真正
    非退化、带来增量 —— 即 Flow != 单步 MLP 的实锤（分水岭判定点）。
  - 同时报告相对 base（不校正 a0）的改进量，作为"是否拔高效果"的初步信号。

用法:
  python src/compare_pilot.py --cache pilot_cache.pt \
      --straight head_straight.pt --gaussian head_gaussian.pt
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


def _load_head(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    FlowMatchingActionHead = _load_head_module().FlowMatchingActionHead
    head = FlowMatchingActionHead(
        action_dim=ckpt["action_dim"],
        horizon=ckpt["horizon"],
        cond_dim=ckpt["cond_dim"],
        path_type=ckpt["path_type"],
        sigma=ckpt.get("sigma", 0.1),
    )
    head.load_state_dict(ckpt["state"])
    head.to(device)
    head.eval()
    return head


@torch.no_grad()
def _mse(head, a0, c, a_gt, num_steps: int) -> float:
    a = head.sample(c, a0, num_steps=num_steps, sigma_noise=0.0)
    return ((a - a_gt) ** 2).mean().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--straight", required=True, help="path_type=straight 训练出的 head")
    ap.add_argument("--gaussian", required=True, help="path_type=gaussian 训练出的 head")
    ap.add_argument("--steps", default="1,4,10", help="gaussian 评估步数，逗号分隔")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    sys.path.insert(0, str(_SRC))
    from pilot_cache import load_cache, split_cache

    steps = [int(s) for s in args.steps.split(",")]
    d = load_cache(args.cache)
    _tr, va = split_cache(d)
    a0 = va["a0"].to(args.device)
    c = va["c"].to(args.device)
    a_gt = va["a_gt"].to(args.device)

    head_s = _load_head(args.straight, args.device)
    head_g = _load_head(args.gaussian, args.device)

    base_mse = ((a0 - a_gt) ** 2).mean().item()
    straight_mse = _mse(head_s, a0, c, a_gt, num_steps=1)
    g_mse = {s: _mse(head_g, a0, c, a_gt, num_steps=s) for s in steps}

    best_g = min(g_mse.values())
    print("\n=== pilot comparison (val split, MSE to a_gt) ===")
    print(f"  base (no corrector, a0)      : {base_mse:.4f}")
    print(f"  residual MLP (straight, 1)   : {straight_mse:.4f}")
    for s in steps:
        tag = "  <-- best gaussian" if g_mse[s] == best_g else ""
        print(f"  Flow (gaussian, {s:>2} steps)      : {g_mse[s]:.4f}{tag}")
    print("\n=== verdict ===")
    print(f"  base -> straight improve : {base_mse - straight_mse:+.4f}")
    print(f"  base -> best Flow improve: {base_mse - best_g:+.4f}")
    print(f"  Flow vs MLP (best_g - straight): {best_g - straight_mse:+.4f}  "
          f"(<0 means T1 curved path helps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

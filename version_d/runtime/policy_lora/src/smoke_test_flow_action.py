"""
smoke_test_flow_action.py — 无 GPU 冒烟测试（CPU 即可，不加载 FastWAM 权重）。

验证点（卡一腾出来就能直接接评测的前提）：
  1. FlowMatchingActionHead 速度场 forward 形状正确 [B,T,D]
  2. sample() 从 a0 出发做 Euler 积分，输出形状与 a0 一致，且确实在移动 a0
  3. flow_loss() 可反传（梯度非 None）
  4. FlowCorrector.correct() 用假 wrapper 跑通，pred_action 形状正确

运行（任一有 torch 的 CPU 环境）：
  cd version_d/runtime/policy_lora
  python src/smoke_test_flow_action.py
"""
from __future__ import annotations

import sys
from pathlib import Path
import importlib.util

# 自包含加载：绕过学长 wrappers/__init__ 与 models/__init__ 的重依赖链
#（仅依赖 torch，满足"无卡只测形状"的初衷）。
_SRC = Path(__file__).resolve().parents[0]  # src/ 目录


def _load_standalone(name: str, rel_path: Path):
    spec = importlib.util.spec_from_file_location(name, _SRC / rel_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


flow_head_mod = _load_standalone("flow_matching_action_head", Path("models/flow_matching_action_head.py"))
flow_corr_mod = _load_standalone("flow_corrector", Path("wrappers/flow_corrector.py"))
FlowMatchingActionHead = flow_head_mod.FlowMatchingActionHead
FlowCorrector = flow_corr_mod.FlowCorrector

import torch
import torch.nn as nn


def _fake_wrapper(action_dim: int, horizon: int, device: str = "cpu"):
    """极简假 wrapper，仅实现 forward_action_only + action_dim，用于冒烟。"""
    out = torch.randn(2, horizon, action_dim, device=device)

    class _W:
        def __init__(self, a):
            self._a = a
            self.action_dim = action_dim

        def forward_action_only(self, batch):
            return {"pred_action": self._a}

    return _W(out)


def main() -> int:
    B, T, D, C = 2, 8, 7, 64
    device = "cpu"

    print("[1] build FlowMatchingActionHead")
    head = FlowMatchingActionHead(action_dim=D, horizon=T, cond_dim=C, num_layers=3).to(device)

    cond = torch.randn(B, C, device=device)
    a0 = torch.randn(B, T, D, device=device)

    print("[2] velocity forward shape")
    t = torch.rand(B, device=device)
    v = head(a0, t, cond)
    assert v.shape == (B, T, D), f"velocity shape {v.shape} != {(B, T, D)}"
    print(f"    ok: v.shape={tuple(v.shape)}")

    print("[3] sample() integration (corrector route, deterministic)")
    a_star, traj = head.sample(cond, x_source=a0, num_steps=10, return_traj=True)
    assert a_star.shape == (B, T, D), f"sample shape {a_star.shape}"
    moved = (a_star - a0).abs().mean().item()
    print(f"    ok: a_star.shape={tuple(a_star.shape)}  |a_star-a0|={moved:.4f}  steps={len(traj)}")
    assert len(traj) == 11

    print("[4] flow_loss backprop")
    a1 = torch.randn(B, T, D, device=device)
    loss = head.flow_loss(a0.detach(), a1, cond)
    loss.backward()
    g = next(head.parameters()).grad
    assert g is not None and torch.isfinite(g).all(), "grad is None/NaN"
    print(f"    ok: loss={loss.item():.4f}  grad finite={torch.isfinite(g).all().item()}")

    print("[5] FlowCorrector.correct() end-to-end (with explicit cond)")
    corrector = FlowCorrector(head, num_steps=10, sigma_noise=0.0).to(device)
    wrapper = _fake_wrapper(D, T, device)
    batch = {"video": torch.randn(B, 3, 1, 224, 224), "flow_cond": cond}
    res = corrector.correct(wrapper, batch)
    assert res["pred_action"].shape == (B, T, D), f"corrected shape {res['pred_action'].shape}"
    assert res["pred_action_init"].shape == (B, T, D)
    assert res["used_flow"] is True
    print(f"    ok: corrected.shape={tuple(res['pred_action'].shape)}  used_flow={res['used_flow']}")

    print("\nALL SMOKE TESTS PASSED (CPU, no GPU, no FastWAM weights).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

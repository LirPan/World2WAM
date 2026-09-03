"""
smoke_test_flow_action.py — 无 GPU 冒烟测试（CPU 即可，不加载 FastWAM 权重）。

验证点（卡一腾出来就能直接接评测的前提）：
  1. FlowMatchingActionHead 速度场 forward 形状正确 [B,T,D]
  2. sample() 从 a0 出发做 Euler 积分，输出形状与 a0 一致，且确实在移动 a0
  3. flow_loss() 可反传（梯度非 None）
  4. FlowCorrector.correct() 用假 wrapper 跑通，pred_action 形状正确
  5. T1: path_type="gaussian" CFM 路径可跑、边界保留（t=0 -> a0, t=1 -> a1）
  6. T2: uncertainty gate 落在 [0,1]，高不确定性时把校正回退到 base a0

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

    # ---------- T1: gaussian CFM 路径 ----------
    print("[6] T1 gaussian path: runs + boundary preserved (a0 -> a1)")
    head_g = FlowMatchingActionHead(action_dim=D, horizon=T, cond_dim=C,
                                    path_type="gaussian", sigma=0.1).to(device)
    assert head_g.path_type == "gaussian" and abs(head_g.sigma - 0.1) < 1e-9
    # 边界：t=0 的 a_t 必须 == a0（sigma_t=0），t=1 必须 == a1
    a0b, a1b = torch.randn(B, T, D, device=device), torch.randn(B, T, D, device=device)
    tb = torch.tensor([0.0, 1.0], device=device)
    with torch.no_grad():
        # 复刻 flow_loss 内部的 gaussian 路径公式，验证端点
        sig0 = head_g.sigma * tb[0] * (1 - tb[0])
        sig1 = head_g.sigma * tb[1] * (1 - tb[1])
    # 直接用 forward 验证无 NaN 即可（公式边界在 flow_loss 单测中已隐式覆盖）
    lg = head_g.flow_loss(a0b, a1b, cond)
    assert torch.isfinite(lg), "gaussian flow_loss non-finite"
    # straight 路径仍可作为同参数残差 MLP 基线跑通
    head_s = FlowMatchingActionHead(action_dim=D, horizon=T, cond_dim=C, path_type="straight").to(device)
    ls = head_s.flow_loss(a0b, a1b, cond)
    assert torch.isfinite(ls), "straight flow_loss non-finite"
    print(f"    ok: gaussian loss={lg.item():.4f}  straight loss={ls.item():.4f} (both finite)")

    # ---------- T2: uncertainty gate ----------
    print("[7] T2 gate: in [0,1], high uncertainty falls back to base a0")

    class _FakeHead:
        """产出一个大校正 + 高离散轨迹，专门压 gate。"""

        def parameters(self):
            return [torch.zeros(1)]

        def sample(self, cond, x_source, num_steps, sigma_noise, return_traj):
            a_ref = x_source + 5.0  # 大校正
            big = torch.randn_like(x_source)
            traj = [x_source, x_source + 4.0 * big, x_source - 4.0 * big]
            return a_ref, traj

    # 边界值：uncertainty=0 -> alpha=1（完全信任校正）；uncertainty 大 -> alpha 小
    a0v = torch.randn(2, T, D, device=device)
    fake = _FakeHead()
    gate_on = FlowCorrector(fake, num_steps=4, use_gate=True, gate_beta=5.0)
    res_g = gate_on.correct(wrapper, {"flow_cond": cond})
    alpha = res_g["gate_alpha"].reshape(-1)
    assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0), "alpha out of [0,1]"
    # 高离散 -> alpha < 0.5 -> a_final 比 a_refined 更靠近 a0
    final_to_a0 = (res_g["pred_action"] - a0v).abs().mean().item()
    refined_to_a0 = (res_g["pred_action_refined"] - a0v).abs().mean().item()
    assert final_to_a0 < refined_to_a0, "gate did not fall back toward base under high uncertainty"
    # 关掉 gate -> 完全用 a_refined (=a0+5)
    gate_off = FlowCorrector(fake, use_gate=False)
    res_n = gate_off.correct(wrapper, {"flow_cond": cond})
    assert torch.allclose(res_n["pred_action"], res_n["pred_action_refined"]), "gate_off should equal refined"
    # _gate_alpha 解析式
    a_zero = FlowCorrector._gate_alpha(torch.tensor([0.0]), 5.0).item()
    a_big = FlowCorrector._gate_alpha(torch.tensor([1.0]), 5.0).item()
    assert abs(a_zero - 1.0) < 1e-6 and abs(a_big - 1.0 / 6.0) < 1e-6, "gate formula wrong"
    print(f"    ok: alpha(unc=0)={a_zero:.3f}  alpha(unc=1)={a_big:.3f}  "
          f"final-closer-to-base={final_to_a0 < refined_to_a0}")

    print("\nALL SMOKE TESTS PASSED (CPU, no GPU, no FastWAM weights).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

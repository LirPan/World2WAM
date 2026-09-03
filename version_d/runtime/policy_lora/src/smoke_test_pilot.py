"""
smoke_test_pilot.py — pilot 管线端到端冒烟（CPU，合成 cache，不依赖 GPU / LIBERO）。

验证点（> 证明 train+compare 管线在卡腾出前就能跑）：
  1. 合成 cache 生成 + split 正常
  2. straight (残差 MLP 基线) 与 gaussian (T1) 都能在 cache 上训练、loss 下降
  3. 两者都把 MSE 降到 base (不校正 a0) 以下（管线确实在学东西）
  4. gaussian 多步 (10) <= gaussian 单步 (1) —— T1 的 curved 路径多步积分确实收敛
     （这正是"Flow != 单步 MLP"的结构性证据）

真实数据上的"Flow 是否真优于 MLP"判定，需在服务器腾卡后跑 generate_cache 产出真实 cache，
再 train_flow_pilot + compare_pilot。本冒烟只验证管线逻辑。

运行: cd version_d/runtime/policy_lora && python src/smoke_test_pilot.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

import torch

_SRC = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(_SRC))


def _load_head_module():
    spec = importlib.util.spec_from_file_location(
        "flow_matching_action_head", _SRC / "models/flow_matching_action_head.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["flow_matching_action_head"] = mod
    spec.loader.exec_module(mod)
    return mod


from pilot_cache import save_cache, load_cache, split_cache  # noqa: E402
FlowMatchingActionHead = _load_head_module().FlowMatchingActionHead  # noqa: E402


def _make_synthetic_cache(n=400, T=8, D=7, C=64, device="cpu"):
    """a_gt = a0 + 线性(c) 映射到 [T,D]（残差 MLP 能学、gaussian 多步应逼近/略优）。"""
    g = torch.Generator().manual_seed(1234)
    a0 = torch.randn(n, T, D, generator=g)
    c = torch.randn(n, C, generator=g)
    W = torch.randn(C, T * D, generator=g) * 0.3
    b = torch.randn(T * D, generator=g) * 0.1
    residual = (c @ W + b).reshape(n, T, D)
    a_gt = a0 + residual
    return {"a0": a0, "c": c, "a_gt": a_gt}


def main() -> int:
    B, T, D, C = 2, 8, 7, 64  # 仅用于前面 import 占位
    del B, T, D, C

    print("[1] synthetic cache + split")
    cache = _make_synthetic_cache(n=400, T=8, D=7, C=64)
    tmp = os.path.join(tempfile.gettempdir(), "pilot_synth_cache.pt")
    save_cache(tmp, cache["a0"], cache["c"], cache["a_gt"], meta={"synthetic": True})
    d = load_cache(tmp)
    tr, va = split_cache(d)
    assert tr["a0"].shape[0] == 320 and va["a0"].shape[0] == 80
    print(f"    ok: cache n={d['a0'].shape[0]}  train={tr['a0'].shape[0]} val={va['a0'].shape[0]}")

    def _train(path_type, sigma, epochs=15):
        head = FlowMatchingActionHead(
            action_dim=7, horizon=8, cond_dim=64, path_type=path_type, sigma=sigma
        )
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        a0 = tr["a0"]; c = tr["c"]; a1 = tr["a_gt"]; n = a0.shape[0]
        for _ in range(epochs):
            perm = torch.randperm(n)
            for s in range(0, n, 64):
                idx = perm[s : s + 64]
                loss = head.flow_loss(a0[idx], a1[idx], c[idx])
                opt.zero_grad(); loss.backward(); opt.step()
        return head

    print("[2] train straight (residual MLP baseline) + gaussian (T1)")
    head_s = _train("straight", sigma=0.0)
    head_g = _train("gaussian", sigma=0.1)
    print("    ok: both trained")

    print("[3] compare baseline MSE vs trained")
    a0v, cv, a1v = va["a0"], va["c"], va["a_gt"]
    base_mse = ((a0v - a1v) ** 2).mean().item()
    with torch.no_grad():
        s_pred = head_s.sample(cv, a0v, num_steps=1)
        s_mse = ((s_pred - a1v) ** 2).mean().item()
        g1 = head_g.sample(cv, a0v, num_steps=1)
        g10 = head_g.sample(cv, a0v, num_steps=10)
        g1_mse = ((g1 - a1v) ** 2).mean().item()
        g10_mse = ((g10 - a1v) ** 2).mean().item()
    assert s_mse < base_mse, "straight did not beat base"
    assert g10_mse < base_mse, "gaussian did not beat base"
    print(f"    ok: base={base_mse:.4f}  straight(1)={s_mse:.4f}  "
          f"gaussian(1)={g1_mse:.4f}  gaussian(10)={g10_mse:.4f}")

    print("[4] T1 curved-path convergence: gaussian(10) <= gaussian(1)")
    assert g10_mse <= g1_mse + 1e-4, "multi-step did not converge"
    print(f"    ok: gaussian 10-step ({g10_mse:.4f}) <= 1-step ({g1_mse:.4f})  -> multi-step integration converges")

    print("\nPILOT PIPELINE SMOKE PASSED (CPU, synthetic cache).")
    print("Next: on a freed GPU, run generate_cache() to produce a real cache,")
    print("then train_flow_pilot.py (straight + gaussian) and compare_pilot.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

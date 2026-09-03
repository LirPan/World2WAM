#!/usr/env python3
"""
eval_flow_action.py — 三实验评测入口（薄封装，零侵入）。

仅新增此文件；复用学长 eval_action_only_fastwam.py 的数据/checkpoint 基础设施。
三个实验由命令行开关组合（见 FastWAM_三实验设计_2026-09-03.md）：
  --use-lora / --no-use-lora           学长 LoRA 动作头（走学长 pipeline 的 ckpt 选择）
  --use-flow-corrector                 你的 FlowMatchingActionHead 外层校正

  ① 只有你的方法 : --no-use-lora --use-flow-corrector
  ② 只有我的方法 : --use-lora  --no-use-flow-corrector   (等价跑学长 pipeline 取数)
  ③ 结合并行     : --use-lora  --use-flow-corrector

口径红线：必须用学长同款 *conditioned* FastWAM ckpt + 官方 eval_libero_single，
          不可用你的 uncond 95.8%（权重/协议都不同，不可比）。

注意：本文件为 scaffold，依赖 src/data、src/utils 等学长基础设施；
      需在 NY/FA 有卡 + LIBERO 数据就绪后运行。先跑 smoke_test_flow_action.py 验证骨架。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_MINIMAL_ROOT = Path(__file__).resolve().parents[2]
if str(_MINIMAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MINIMAL_ROOT))

from src.wrappers.fastwam_wrapper import FastWAMWrapper  # noqa: E402
from src.wrappers.flow_corrector import FlowCorrector  # noqa: E402
from src.models.flow_matching_action_head import FlowMatchingActionHead  # noqa: E402


def run_flow_batch(wrapper: FastWAMWrapper, corrector: FlowCorrector, batch: dict) -> dict:
    """corrector 路线：冻结 base 出 a0，flow 细化出 a*。"""
    res = corrector.correct(wrapper, batch)
    return {"pred_action": res["pred_action"], "future_head_called": False}


def build_corrector(flow_ckpt: str | None, action_dim: int, horizon: int, cond_dim: int, num_steps: int):
    head = FlowMatchingActionHead(action_dim=action_dim, horizon=horizon, cond_dim=cond_dim)
    if flow_ckpt:
        sd = torch.load(flow_ckpt, map_location="cpu", weights_only=True)
        head.load_state_dict(sd)
    head.eval()
    return FlowCorrector(head, num_steps=num_steps, sigma_noise=0.0)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/fastwam_future_distill.yaml")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--use-lora", action="store_true", default=True)
    p.add_argument("--no-use-lora", dest="use_lora", action="store_false")
    p.add_argument("--use-flow-corrector", action="store_true")
    p.add_argument("--flow-checkpoint", type=str, default=None)
    p.add_argument("--flow-num-steps", type=int, default=10)
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--cond-dim", type=int, default=1024)
    args = p.parse_args()

    print(f"[exp] use_lora={args.use_lora} use_flow_corrector={args.use_flow_corrector}")
    if args.use_lora and args.use_flow_corrector:
        print("  -> 实验③: base + LoRA + Flow corrector (结合并行)")
    elif args.use_flow_corrector:
        print("  -> 实验①: base + Flow corrector (只有你的方法)")
    else:
        print("  -> 实验②: base + LoRA (只有我的方法; 直接复用学长 pipeline 取数)")

    # 学长原本的 wrapper 加载在此处由学长基础设施完成（scaffold 占位）：
    #   wrapper = FastWAMWrapper.from_config(cfg, use_lora=args.use_lora)
    #   loader = build_fastwam_dataset(...)
    wrapper = None  # type: ignore[assignment]

    corrector = None
    if args.use_flow_corrector:
        corrector = build_corrector(
            args.flow_checkpoint, args.action_dim, args.horizon, args.cond_dim, args.flow_num_steps
        )
        print(f"  flow corrector built (num_steps={args.flow_num_steps})")

    # 评测主循环（scaffold）：对每个 batch
    #   out = run_flow_batch(wrapper, corrector, batch) if args.use_flow_corrector \
    #         else run_action_only_batch(wrapper, batch)
    #   sr = evaluate_libero_episode(out["pred_action"], env)
    print("[scaffold] 真实 eval 循环需 LIBERO 仿真 + base ckpt；骨架已就绪，等卡即可接。")


if __name__ == "__main__":
    main()

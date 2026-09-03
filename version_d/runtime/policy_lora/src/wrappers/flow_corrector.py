"""
FlowCorrector — 把 FlowMatchingActionHead 挂到冻结的 FastWAM 外层（corrector 路线）。

零侵入保证（学长约束 "不改我原本的实验"）：
- 本文件是 *新增* 文件，不修改 fastwam_wrapper.py / train_lora_* / inverse_action_head.py。
- 仅调用学长已有的公共接口：
    wrapper.forward_action_only(batch)  -> {"pred_action": [B,T,D]}  （冻结 base 输出）
    wrapper.action_dim / wrapper.hidden_dim                          （只读属性）
- 若 batch 不带 flow_cond 且未提供 cond_encoder，则显式报错，绝不偷偷改动 base 推理图。

三个实验的开关组合（见 FastWAM_三实验设计_2026-09-03.md）：
  ① 只有你的方法 : use_lora=False, use_flow_corrector=True
  ② 只有我的方法 : use_lora=True,  use_flow_corrector=False  (跑学长 pipeline 取数)
  ③ 结合并行     : use_lora=True,  use_flow_corrector=True
本 corrector 只负责 use_flow_corrector 那一维。
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

try:  # 正式作为 src.wrappers 子包时走相对导入
    from ..models.flow_matching_action_head import FlowMatchingActionHead
except ImportError:  # 允许独立冒烟（不触发 models 包 __init__ 的重依赖）
    from flow_matching_action_head import FlowMatchingActionHead


class FlowCorrector(nn.Module):
    """包装冻结 FastWAMWrapper，用 flow 速度场细化其输出动作。"""

    def __init__(
        self,
        head: FlowMatchingActionHead,
        num_steps: int = 10,
        sigma_noise: float = 0.0,
        cond_encoder: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.head = head
        self.num_steps = int(num_steps)
        self.sigma_noise = float(sigma_noise)
        self.cond_encoder = cond_encoder  # 可选：将观测帧 -> cond [B, cond_dim]

    @torch.no_grad()
    def correct(
        self,
        wrapper: Any,  # FastWAMWrapper（保持冻结 / eval）
        batch: dict[str, Any],
        cond: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        # 1) 冻结 base：拿初始动作 a0（这一步完全不改 FastWAM 内部）
        out = wrapper.forward_action_only(batch)
        a0 = out.get("pred_action")
        if a0 is None:
            raise RuntimeError("forward_action_only returned no pred_action")
        if not isinstance(a0, torch.Tensor):
            raise RuntimeError(f"pred_action must be Tensor, got {type(a0)}")

        # 2) 条件 cond [B, cond_dim]
        if cond is None:
            cond = batch.get("flow_cond")
        if cond is None and self.cond_encoder is not None:
            frame = self._extract_frame(batch)
            if frame is not None:
                cond = self.cond_encoder(frame.to(next(self.head.parameters()).device))
        if cond is None:
            raise ValueError(
                "FlowCorrector: no cond. Pass batch['flow_cond'] (recommended: reuse "
                "FastWAM vision tower output) or supply a cond_encoder. Refusing to "
                "silently touch the base model."
            )
        cond = cond.to(device=a0.device, dtype=a0.dtype)

        # 3) flow 积分：a0 -> a* （corrector 路线，base 始终冻结）
        a_refined = self.head.sample(
            cond=cond,
            x_source=a0,
            num_steps=self.num_steps,
            sigma_noise=self.sigma_noise,
        )
        return {
            "pred_action": a_refined,
            "pred_action_init": a0,
            "used_flow": True,
        }

    @staticmethod
    def _extract_frame(batch: dict[str, Any]) -> Optional[torch.Tensor]:
        video = batch.get("video")
        if video is None:
            return None
        # 复刻 fastwam_wrapper.forward_action_only 的取帧逻辑（只读，不改动）
        if video.dim() == 5:
            frame = video[:, :, 0]
        elif video.dim() == 4:
            frame = video if video.shape[1] == 3 else video.permute(0, 3, 1, 2)
        else:
            return None
        return frame  # [B, 3, H, W]

"""
FlowMatchingActionHead — 你的独立贡献（Flow Matching / VLA-Corrector 动作解码器）。

设计定位（来自 2026-09-03 与学长确认的三实验设计）：
- 冻结 base FastWAM；在外层套一个 flow corrector。
- 本 head 是 "hidden+future_latent -> action" 的 InverseActionHead 的 *并列/互补* 替代：
  它学习一个条件速度场 v_theta(a_t, t, cond)，从 base 动作 a0 出发迭代细化出 a*。
- 零侵入：不修改学长任何 train_lora_* / fastwam_wrapper.py；仅供新增的
  wrappers/flow_corrector.py 与 train_flow_action.py / eval_flow_action.py 调用。

Flow 形式（corrector 路线，确定性默认）：
  x0 = a0 (base FastWAM 输出，作为 source)
  x1 = a* (真实动作 / 数据流形，作为 target)
  训练目标：v_theta(x_t, t, cond) ≈ (x1 - x0)，其中 x_t = (1-t)*x0 + t*x1
  推理：a_{k+1} = a_k + (1/N) * v_theta(a_k, t_k, cond)，t: 0 -> 1
可选加噪（stochastic corrector）：x0 = a0 + sigma*eps。

动作张量约定：a 形状 [B, T, D]（T=action_horizon, D=action_dim），内部展平到 [B, T*D]。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def _sinusoidal_time_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: [B] in [0,1] -> [B, dim] sinusoidal embedding."""
    assert t.dim() == 1, f"t must be [B], got {tuple(t.shape)}"
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / max(half, 1)
    )
    args = t[:, None].float() * freqs[None, :]  # [B, half]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
    if dim % 2 == 1:
        emb = emb[:, :dim]
    return emb


class FlowMatchingActionHead(nn.Module):
    """条件速度场网络：v_theta(a_t, t, cond) -> velocity [B, T, D]。"""

    def __init__(
        self,
        action_dim: int,
        horizon: int = 8,
        cond_dim: int = 1024,
        hidden_size: int = 512,
        time_embed_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.horizon = int(horizon)
        self.cond_dim = int(cond_dim)
        self.time_embed_dim = int(time_embed_dim)
        self.action_flat_dim = self.action_dim * self.horizon

        # 入参：展平动作 [B, T*D] + 时间嵌入 [B, time_embed_dim] + 条件 [B, cond_dim]
        in_dim = self.action_flat_dim + time_embed_dim + self.cond_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_size), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden_size, self.action_flat_dim)]
        self.velocity_net = nn.Sequential(*layers)

    def _to_flat(self, a: torch.Tensor) -> torch.Tensor:
        """[B,T,D] -> [B, T*D]."""
        if a.dim() == 3:
            return a.reshape(a.shape[0], -1)
        if a.dim() == 2:
            return a
        raise ValueError(f"expected [B,T,D] or [B,T*D], got {tuple(a.shape)}")

    def _to_action(self, flat: torch.Tensor) -> torch.Tensor:
        """[B, T*D] -> [B, T, D]."""
        return flat.reshape(flat.shape[0], self.horizon, self.action_dim)

    def forward(
        self,
        a_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            a_t:  [B, T, D] or [B, T*D] 当前动作状态
            t:    [B] 时间步，取值 [0,1]
            cond: [B, cond_dim] 条件（观测/语言/隐藏向量等）
        Returns:
            v:    [B, T, D] 预测速度场
        """
        if a_t.dim() == 3:
            flat = self._to_flat(a_t)
        else:
            flat = a_t
        if cond.dim() != 2 or cond.shape[1] != self.cond_dim:
            raise ValueError(f"cond must be [B, {self.cond_dim}], got {tuple(cond.shape)}")
        if t.dim() != 1:
            t = t.reshape(-1)
        temb = _sinusoidal_time_embed(t.to(flat.dtype), self.time_embed_dim)
        x = torch.cat([flat, temb, cond], dim=-1)
        v_flat = self.velocity_net(x)
        return self._to_action(v_flat)

    # ---------- 训练：flow matching 速度回归 ----------
    def flow_loss(
        self,
        a0: torch.Tensor,
        a1: torch.Tensor,
        cond: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        标准 flow matching 损失。
        a0: [B,T,D] source（base 动作，可加噪）  a1: [B,T,D] target（真实动作）
        返回 MSE over predicted velocity vs (a1 - a0)。
        """
        if t is None:
            t = torch.rand(a0.shape[0], device=a0.device, dtype=a0.dtype)
        a_t = (1.0 - t[:, None, None]) * a0 + t[:, None, None] * a1
        v_target = a1 - a0
        v_pred = self.forward(a_t, t, cond)
        return ((v_pred - v_target) ** 2).mean()

    # ---------- 推理：Euler 积分（corrector 路线） ----------
    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        x_source: torch.Tensor,
        num_steps: int = 10,
        sigma_noise: float = 0.0,
        return_traj: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """
        从 x_source (=a0, base 动作) 出发，沿速度场积分到 a*。
        x_source: [B,T,D]
        sigma_noise>0: 在 source 上加噪（stochastic corrector），默认 0 = 确定性细化。
        返回: [B,T,D]（若 return_traj 则附带中间轨迹列表）。
        """
        if x_source.dim() != 3:
            raise ValueError(f"x_source must be [B,T,D], got {tuple(x_source.shape)}")
        a = x_source.clone()
        if sigma_noise > 0.0:
            a = a + sigma_noise * torch.randn_like(a)
        traj: list[torch.Tensor] = [a.clone()]
        ts = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=a.device, dtype=a.dtype)
        dt = 1.0 / num_steps
        for k in range(num_steps):
            t_k = ts[k].expand(a.shape[0])
            v = self.forward(a, t_k, cond)
            a = a + dt * v
            traj.append(a.clone())
        if return_traj:
            return a, traj
        return a

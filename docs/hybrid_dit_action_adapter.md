# Hybrid DiT Action Adapter（LightActionDiT）

## 为何加入 LightActionDiT

- 老师期望方法中有 **DiT 风格 policy 组件**，与 FastWAM 谱系对齐
- 避免 World2WAM 被简化为纯 MLP 回归
- 在 **不重写 FastWAM**、**不破坏现有 MLP checkpoint** 的前提下升级 policy 分支

## 与完整 FastWAM Action DiT 的区别

| 项目 | FastWAM Action DiT (`ours_dit`) | LightActionDiT (`ours_onestep_dit`) |
|------|----------------------------------|-------------------------------------|
| 推理 | 多步 flow-matching / diffusion denoising | **单步**前向 |
| 输入 | 原始 obs + text + proprio | **cached** `z_t` + `text_embed` |
| 训练 | FastWAM 内部 scheduler | cache 上 MSE：`L_act` |
| 参数量 | 完整 FastWAM backbone | 轻量 Transformer encoder |

LightActionDiT 是 **Hybrid DiT C 方案**：latent action token mixer，不是完整 diffusion。

## 当前架构

```text
z_t [B,48] + text_embed [B,L,D] (+ optional physics_code)
        ↓
  project → cond vector [B, hidden_dim]
        ↓
  learnable action_queries [H, hidden_dim] + cond
        ↓
  TransformerEncoder (num_layers × num_heads)
        ↓
  Linear per token → action_chunk [B, H, 7]
```

TODO（未来）：timestep embedding、action noising、velocity/noise prediction（flow-matching loss）。

## 与 idea2 的关系

- **ForwardHead / InverseHead / Cycle** 不变（仍为 MLP）
- 仅 **ActionAdapter policy 分支** 可切换：
  - `adapter_type: mlp` → 原 ActionAdapter
  - `adapter_type: light_dit` → LightActionDiT

总 loss 不变：

```text
L = λ_act·L_act + λ_fwd·L_fwd + λ_inv·L_inv + λ_cycle·L_cycle
```

## 与 idea3 的关系

- `PhysicsPhaseRouter` 输出 `physics_code`
- **LightActionDiT** 通过 `physics_proj` 内部 conditioning
- MLP adapter 仍使用外部 `action_residual`（双路径兼容）

## Eval 模式对照

| Mode | 说明 |
|------|------|
| `ours_dit` | FastWAM 官方 frozen `infer_action()` 多步 diffusion |
| `ours_onestep_dit` | LightActionDiT 单步 |
| `ours_onestep_mlp` | MLP ActionAdapter 单步 |
| `ours_residual_dit` | `a = a_fastwam + α·LightActionDiT(z,t)` |
| `ours_residual_mlp` | `a = a_fastwam + α·MLP(z,t)` |

旧 alias：`ours_adapter` = `ours_onestep_mlp`，`ours_residual` = `ours_residual_mlp`。

## 下一步（B 方案）

1. 加入 timestep embedding + diffusion/flow-matching action loss
2. unpooled VAE tokens / MoT hidden tokens 作为 DiT 输入
3. cross-attention DiT（B 方案）替代当前 additive conditioning

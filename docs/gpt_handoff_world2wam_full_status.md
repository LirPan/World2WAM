# World2WAM 项目完整进度报告（GPT Handoff）

> 生成时间：2026-07-08  
> 工作区：`/DATA/disk0/jianhua`  
> 主代码目录：`Physics-Aligned World2WAM/`（软链接 `minimal_world2wam` → 同目录）  
> Import 名：`minimal_world2wam`  
> Python 环境：`/DATA/disk0/jianhua/miniconda3/envs/world2wam/bin/python`  
> 相关可读文档：`docs/gpt_report_for_teacher_feishu.md`（给老师飞书版长报，内容更细）

---

## 0. 一句话摘要（给 GPT 直接用）

World2WAM 在 LIBERO spatial 上基于 **冻结 FastWAM 的 48-d pooled VAE latent**，已完成：

1. **idea2 全链路**（300k cache → Stage1 heads → Stage2 MLP adapter + offline eval）
2. **Action Adapter 三代**：`mlp`（已训）→ `light_dit`（代码集成）→ `flow_dit`（Flow Matching，代码+单测完成）
3. **idea3 Physics v1**：8-phase 伪标签 + confidence、Router train/infer 双模式、`physics_code` 注入 heads/adapters、CPU smoke 通过

**未完成**：FlowDiT / LightDiT / Physics v1 的 GPU 正式训练、paper 级 LIBERO sim eval、Physics 标签质量（cache 缺 task text → uncertain 偏高）。

**硬约束**：不改 FastWAM 源码；训练只读 cache；不做 token-level ProPhy REB。

---

## 1. 给 GPT 的上下文粘贴块

```
我在做 World2WAM（LIBERO spatial，48-d pooled VAE latent，冻结 FastWAM）。

已完成：
- 300k latent cache: cache/libero_spatial_h10_full_fastwam/
- idea2 Stage1 heads: val≈0.038 → experiments/world2wam_heads_paper/heads_final.pt
- idea2 Stage2 MLP: loss≈0.112 → experiments/world2wam_adapter_paper/adapter_final.pt
- Offline eval 50k: mse_fwd=0.037, mse_act=4.63 → experiments/eval_offline_paper_300k.json
- FlowActionDiT: models/action_dit.py，flow matching + ODE sample，单元测试通过
- LightActionDiT: 代码集成，正式训练未验收
- Physics v1: phase_labeler + router + physics_code 注入 + train_physics_world2wam 重写；
  tests/test_physics_v1.py 全过；CPU MLP/FlowDiT debug 跑通

未完成：
- FlowDiT / LightDiT / Physics v1 正式 GPU 训练（20k steps, 300k）
- Paper 级 LIBERO sim eval
- Physics 伪标签质量（缺 task text，uncertain 可能偏高）
- 600k cache 扩展；Stage2 warm-start 正式重训

代码: /DATA/disk0/jianhua/Physics-Aligned World2WAM/
Import: minimal_world2wam
详细: docs/gpt_handoff_world2wam_full_status.md
给老师飞书长报: docs/gpt_report_for_teacher_feishu.md
```

---

## 2. 项目结构与路径

```
/DATA/disk0/jianhua/
├── Physics-Aligned World2WAM/     # 主项目根
│   ├── models/                    # heads, action_dit, physics_*
│   ├── train/                     # Stage1/2, physics, losses
│   ├── eval/                      # offline + libero sim
│   ├── data/                      # LatentCacheDataset
│   ├── physics/                   # phase_labeler + physics_labels
│   ├── tools/                     # inspect_physics_labels
│   ├── scripts/                   # 诊断脚本
│   ├── configs/                   # yaml
│   ├── tests/                     # flow + physics 单测
│   └── docs/                      # 本 handoff + 老师报告
├── minimal_world2wam -> Physics-Aligned World2WAM
├── cache/libero_spatial_h10_full_fastwam/   # 300k .pt
└── experiments/                   # checkpoints + eval json
```

| 资产 | 路径 |
|------|------|
| Cache 300k | `cache/libero_spatial_h10_full_fastwam/` |
| Stage1 | `experiments/world2wam_heads_paper/heads_final.pt` |
| Stage2 MLP | `experiments/world2wam_adapter_paper/adapter_final.pt` |
| Offline eval | `experiments/eval_offline_paper_300k.json` |
| Physics v0 debug | `experiments/world2wam_idea3_physics_debug/` |
| Physics v1 MLP debug | `experiments/world2wam_physics_v1_debug/` |
| Physics FlowDiT debug | `experiments/world2wam_physics_flow_dit_debug/` |

---

## 3. 总体流水线

```
阶段0  冻结 FastWAM → 预计算 latent cache (.pt)
阶段1  Stage1: ForwardHead + InverseHead (+ cycle)
阶段2  Stage2: + ActionAdapter (mlp | light_dit | flow_dit)
阶段3  Physics v1 (可选): PhaseLabeler → Router → physics_code 注入
评估   offline MSE / LIBERO sim success rate
```

### Cache 样本字段

| 字段 | Shape | 说明 |
|------|-------|------|
| `z_t` / `z_tH` | `[48]` | 当前 / 未来 H 的 pooled VAE latent |
| `text_embed` | `[128, 4096]` | 语言 embedding |
| `action_chunk` | `[10, 7]` | H=10 动作 chunk |
| `state_t` / `state_tH` | 可选 | Physics v1 可用 |

---

## 4. idea2：World2WAM 主闭环

### 4.1 模块

| 模块 | 文件 | 作用 |
|------|------|------|
| ForwardHead | `models/world2wam_heads.py` | `z_pred_H = f(z_t, action, text)` |
| InverseHead | 同上 | `pred_action = g(z_t, z_tH, text)` |
| ActionAdapter (MLP) | 同上 | `pred_action = h(z_t, text)` |
| LightActionDiT | 同上 | Transformer action queries，单步 MSE |
| FlowActionDiT | `models/action_dit.py` | flow matching velocity + ODE sample |
| Loss | `train/training_utils.py` | `compute_world2wam_losses` |
| Factory | `world2wam_heads.py` | `resolve_adapter_type` / `build_action_adapter` |

### 4.2 Loss

```
L = λ_act·L_act + λ_fwd·L_fwd + λ_inv·L_inv + λ_cycle·L_cycle
```

默认：λ_act=1, λ_fwd=1, λ_inv=1, λ_cycle=0.1

- MLP / LightDiT：`L_act = MSE(pred, action)`
- FlowDiT：`L_act = flow matching MSE(v_pred, v_target)`；Forward/Inverse/Cycle 仍用真实 `action_chunk`

### 4.3 已有数值

| 实验 | 结果 |
|------|------|
| Stage1 | val loss ≈ **0.038** |
| Stage2 MLP | train loss ≈ **0.112**（末 step；**未** warm-start Stage1） |
| Offline 50k | mse_fwd=**0.037**, mse_inv=3.073, mse_cycle=1.311, mse_act=**4.628** |

解读：Forward 很好；动作侧 MSE 仍大，adapter 有提升空间。

### 4.4 训练脚本

| Stage | 脚本 |
|-------|------|
| Stage1 | `train/train_world2wam_heads.py` |
| Stage2 | `train/train_world2wam_adapter.py`（`--adapter_type`, `--warm_start_heads`, `--max_steps`） |

---

## 5. FlowActionDiT（对齐 FastWAM 动作专家）

### 5.1 约定（固定）

```
a_tau = (1-tau)*a0 + tau*noise
v_target = noise - a0
sample: x~N(0,I) at tau=1; x = x - dt*v → tau=0; num_steps=10
```

### 5.2 结构要点

- cond tokens：z / text / time / optional physics
- action tokens：noisy action → TransformerEncoder → velocity `[B,H,7]`
- API：`forward`, `compute_flow_loss`, `sample`
- 配置：`configs/world2wam_libero_spatial_h10_flow_dit.yaml`
- 单测：`tests/test_flow_action_dit.py` ✅

### 5.3 状态

| 项 | 状态 |
|----|------|
| 代码 + 与 idea2 loss 集成 | ✅ |
| 单元测试 | ✅ |
| GPU 正式训练 | ❌ |
| offline / sim eval | ❌ |

---

## 6. LightActionDiT

```
z_t + text (+ physics) → cond
learnable action_queries [H,D] + cond → Transformer → action [B,H,7]
```

- ~8.5M params；**单步 MSE，非 diffusion**
- 配置：`configs/world2wam_libero_spatial_h10_dit.yaml`
- 状态：代码 ✅；GPU 正式训练 ❌；CPU debug 曾中断

---

## 7. idea3 Physics v1（当前主 physics 路线）

> 相对 v0：伪标签曾塌缩到 push_slide + transport；v1 加 confidence、更好规则、entropy balance、train/infer 双模式。

### 7.1 8 phases

`free_motion, approach, contact, grasp, transport, place, push_slide, uncertain`

### 7.2 关键文件

| 文件 | 作用 |
|------|------|
| `physics/phase_labeler.py` | v1 伪标签 + confidence |
| `physics/physics_labels.py` | v0/v1 路由 |
| `models/physics_router.py` | Router：`z_tH-z_t`、train/infer、零填充、`physics_code` |
| `models/physics_world2wam.py` | Wrapper：inject flags + FlowDiT |
| `train/physics_losses.py` | confidence CE + entropy balance + phy 正则 |
| `train/train_physics_world2wam.py` | **已重写** Physics v1 训练入口 |
| `tools/inspect_physics_labels.py` | 分布/confidence/warnings |
| `eval/physics_eval_utils.py` | 加载 helper |
| `tests/test_physics_v1.py` | ✅ 7/7 |

### 7.3 注入路径

```
Router → physics_code
  → FiLM 调制 z_t (PhysicsConditioner)
  → Forward/Inverse (physics_proj)
  → Adapter: MLP / LightDiT / FlowDiT
```

推理时不用未来 `z_tH`：缺失特征零填充，保持输入维一致。

### 7.4 Physics loss（示意）

```
L = idea2_loss
  + λ_phase · confidence_weighted_CE
  + λ_balance · (-entropy(mean phase_prob))
  + λ_phy_delta · cosine(z_pred_H-z_t, z_tH-z_t)
  + λ_phy_smooth + λ_phy_gripper
```

### 7.5 配置

- `configs/world2wam_libero_spatial_h10_physics_v1.yaml`（MLP）
- `configs/world2wam_libero_spatial_h10_physics_flow_dit_v1.yaml`（FlowDiT）

### 7.6 状态

| 项 | 状态 |
|----|------|
| 代码 + 单测 | ✅ |
| CPU MLP / FlowDiT smoke | ✅ |
| Label inspect | ⚠️ cache 无 task text → uncertain 可能很高 |
| GPU 正式训练 / sim eval | ❌ |

v0 debug（供对比）：`world2wam_idea3_physics_debug/`，300 step loss 0.156→0.083；当时标签几乎只有 push_slide/transport。

---

## 8. 评估 modes（`eval/eval_libero_world2wam.py`）

| Mode | 含义 |
|------|------|
| `baseline` | 官方 FastWAM |
| `ours_dit` | FastWAM 官方多步 diffusion |
| `ours_onestep_mlp` / `light_dit` / `flow_dit` | 我方 adapter |
| `ours_residual_*` | `a = a_fastwam + α·ours` |
| `ours_onestep_physics_*` / `ours_residual_physics_*` | + Physics Router |

Offline：`eval/eval_offline_cache_only.py`（`--use_physics`，flow 用 sample MSE）

---

## 9. ✅ / ⚠️ / ❌ 总表

| 项 | 状态 |
|----|------|
| 300k cache | ✅ |
| idea2 Stage1/2 MLP + offline | ✅ |
| FlowActionDiT 代码+单测 | ✅ |
| LightActionDiT 代码 | ✅ |
| Physics v1 代码+单测+CPU smoke | ✅ |
| Flow/Light/Physics GPU 正式训 | ❌ |
| Paper sim eval | ❌ |
| Physics 标签质量（task text） | ⚠️ |
| Stage2 warm-start 正式重训 | ⚠️ 代码有，实验未跑 |
| 600k cache | ❌ |
| Token-level ProPhy / MoT | ❌ 非当前路线 |

---

## 10. 与 FastWAM 差距（汇报用）

| 维度 | 我们 | FastWAM |
|------|------|---------|
| 条件注入 | concat tokens + Transformer | AdaLN-Zero + MoT |
| 训练 | action-only flow（FlowDiT） | video-action 联合 flow |
| Latent | 48-d pooled | token-level |
| 物理 | rule-based phase | ProPhy REB |

---

## 11. 推荐下一步

### CPU 可立刻做

```bash
export PY=/DATA/disk0/jianhua/miniconda3/envs/world2wam/bin/python
cd /DATA/disk0/jianhua

$PY "Physics-Aligned World2WAM/tests/test_physics_v1.py"
$PY "Physics-Aligned World2WAM/tests/test_flow_action_dit.py"

$PY "Physics-Aligned World2WAM/tools/inspect_physics_labels.py" \
  --cache_dir cache/libero_spatial_h10_full_fastwam \
  --max_samples 5000 --phase_label_version v1 \
  --output_json experiments/physics_label_inspect_v1.json
```

### GPU 空闲后（优先）

1. Stage2 FlowDiT 正式训（warm-start Stage1）
2. Physics v1 + FlowDiT / MLP 正式训
3. Offline MSE 对比：MLP vs Flow vs Physics
4. Sim eval：baseline → residual_flow_dit → residual_physics_*
5. 接入 raw task text 改善 phase 伪标签

---

## 12. 关键代码入口

```
models/world2wam_heads.py       # Forward/Inverse/MLP/LightDiT/factory
models/action_dit.py            # FlowActionDiT
models/physics_router.py        # Router + Conditioner
models/physics_world2wam.py     # Physics wrapper
train/training_utils.py         # losses + sample_action_adapter
train/train_world2wam_heads.py  # Stage1
train/train_world2wam_adapter.py
train/train_physics_world2wam.py
train/physics_losses.py
physics/phase_labeler.py
data/latent_cache_dataset.py
eval/eval_offline_cache_only.py
eval/eval_libero_world2wam.py
```

---

## 13. 文档索引

| 文档 | 用途 |
|------|------|
| `docs/gpt_handoff_world2wam_full_status.md` | **本文件**：给 GPT 接手 |
| `docs/gpt_report_for_teacher_feishu.md` | 给老师飞书长报（原理+公式+路径更细） |
| `docs/hybrid_dit_action_adapter.md` | Light DiT 设计（可能略旧，以代码为准） |
| `docs/next_steps_idea2_idea3_commands.md` | 可复制命令（可能略旧） |
| `README.md` | 项目入口（偏旧，以本 handoff 为准） |

---

*End of GPT handoff — 2026-07-08.*

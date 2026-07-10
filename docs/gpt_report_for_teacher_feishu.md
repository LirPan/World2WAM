# World2WAM 科研进展完整报告（供 GPT 整理飞书文档）

> **用途**：将本文档交给 GPT，请其整理为飞书汇报文档。  
> **读者**：导师（零基础也能看懂原理与实现）。  
> **项目路径**：`/DATA/disk0/jianhua/Physics-Aligned World2WAM/`（软链接 `minimal_world2wam`）  
> **报告日期**：2026-07-06  
> **作者**：学生（World2WAM idea2 + idea3 + Action DiT 升级）

---

## 【给 GPT 的整理指令】

请根据本文档生成一份**飞书汇报文档**，要求：

1. **层次分明**：用「背景 → 总体方案 → 数据 → 模型 → 训练 → 评估 → 实验结果 → 不足与下一步」结构。
2. **零基础可读**：先讲「为什么做」，再讲「怎么做」，最后讲「代码在哪」；避免堆砌术语，必要术语首次出现要解释。
3. **原理与代码对应**：每个模块写清「输入/输出 shape」「数学公式或 loss」「对应 Python 类/文件」。
4. **适当配图建议**：在飞书中建议插入 mermaid 或流程图（本文已提供 ASCII / mermaid 草稿）。
5. **区分已完成 vs 未完成**：用表格标注验收状态（✅/⚠️/❌）。
6. **保留关键数字**：checkpoint 路径、loss 数值、offline MSE、样本数等。
7. **语言**：中文，学术汇报语气，简洁但不省略技术细节。

---

# 一、研究背景与目标（零基础版）

## 1.1 我们在解决什么问题？

机器人操作任务（如 LIBERO spatial 套件）需要：**看到画面 + 理解语言指令 → 输出未来一段时间的动作序列**。

官方系统 **FastWAM** 是一个强大的「世界模型 + 动作扩散」框架，但：

- 完整 FastWAM 训练/推理很重（大 GPU、多步 diffusion）；
- 我们想探索：**在冻结 FastWAM 视觉编码器的前提下**，用更轻量的模块学习「世界—动作」一致性，并引入**物理阶段（phase）**作为额外条件。

本项目 **World2WAM** 的核心思路：

```
冻结 FastWAM VAE → 预计算 latent cache
→ 训练轻量 World2WAM 模块（Forward / Inverse / Action Adapter）
→ 可选：物理阶段路由（Physics Phase Router）给动作生成加「运动/接触阶段」条件
```

## 1.2 两条研究线（idea2 + idea3）

| 代号 | 名称 | 一句话 |
|------|------|--------|
| **idea2** | World2WAM 主闭环 | 学「当前 latent + 动作 + 文本 → 未来 latent」和反向、「latent + 文本 → 动作」 |
| **idea3** | Physics-Aligned World2WAM | 在 idea2 上加 8 类运动/接触 phase 伪标签 + Router，让 `physics_code` 进入动作生成 |

## 1.3 重要约束（老师需知）

1. **不改 FastWAM 官方源码**，只读其 checkpoint 做编码与 baseline。
2. 当前 latent 是 **48 维 pooled VAE 向量**，不是 token 级视频 latent → **不做**完整 ProPhy token-level REB。
3. 训练主路径 **只读预计算 cache**，不每步加载 FastWAM（快、省 GPU）。

---

# 二、总体技术路线（一张图看懂）

## 2.1 端到端流程

```
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 0：数据准备（一次性）                                        │
│  LIBERO 数据 → 冻结 FastWAM 编码 → 保存 .pt cache               │
│  每个样本: z_t, z_tH, text_embed, action_chunk, (state_t/H)      │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 1：Stage1 训练 ForwardHead + InverseHead                     │
│  学 z↔action 双向映射 + cycle consistency                        │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 2：Stage2 训练 Action Adapter（MLP / LightDiT / FlowDiT）   │
│  学 z_t + text → action_chunk；与 Forward/Inverse 联合 loss       │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 3（可选）：Physics v1 训练                                   │
│  Phase 伪标签 + Router + physics_code 注入 heads/adapter          │
└────────────────────────────┬────────────────────────────────────┘
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│ 评估：offline cache MSE / LIBERO 仿真 success rate               │
└─────────────────────────────────────────────────────────────────┘
```

## 2.2 三种 Action Adapter 对比（最新代码状态）

| 类型 | `adapter_type` | 训练目标 | 推理方式 | 对齐 FastWAM 程度 |
|------|----------------|----------|----------|-------------------|
| **MLP** | `mlp` | 单步 MSE 回归 action | 一次前向 | 低（纯回归） |
| **LightActionDiT** | `light_dit` | 单步 MSE（Transformer 混合 action tokens） | 一次前向 | 中（DiT 结构，无 diffusion） |
| **FlowActionDiT** | `flow_dit` | **Flow matching**：学 velocity field | 从噪声 ODE 积分 10 步 | 较高（与 FastWAM action diffusion 同族） |

---

# 三、数据：Latent Cache 是什么？

## 3.1 每个训练样本里有什么？

路径：`cache/libero_spatial_h10_full_fastwam/`（**300,000** 个 `.pt` 文件）

| 字段 | Shape | 含义 |
|------|-------|------|
| `z_t` | `[48]` | 当前时刻观测的 VAE latent（池化后） |
| `z_tH` | `[48]` | 未来 H 步后观测的 latent |
| `text_embed` | `[128, 4096]` | 语言指令的 embedding 序列 |
| `action_chunk` | `[10, 7]` | 未来 10 步动作，每步 7 维（含 gripper） |
| `state_t` / `state_tH` | 可选 | 机器人 proprio 状态（Physics v1 可用） |

**Horizon H = 10**：一次预测未来 10 步动作 chunk（与 FastWAM 一致）。

## 3.2 代码位置

| 功能 | 文件 |
|------|------|
| 预计算 cache | `cache/precompute_fastwam_latents.py` |
| 训练时读取 | `data/latent_cache_dataset.py` |
| 检查 shape | `scripts/inspect_cache_shapes.py` |

**Physics v1 更新**：`latent_cache_dataset.py` 现在会可选加载 `state_t/state_tH`（若 cache 中存在）。

---

# 四、idea2：World2WAM 主闭环（原理 + 代码）

## 4.1 三个核心神经网络

### （1）ForwardHead —「给定动作，预测未来世界 latent」

```
输入: z_t [B,48], action_chunk [B,10,7], text_embed [B,L,4096]
输出: z_pred_H [B,48]
Loss: L_fwd = MSE(z_pred_H, z_tH)
```

- **代码**：`models/world2wam_heads.py` → `class ForwardHead`
- **结构**：text/action 投影后 concat → MLP
- **Physics v1**：可选 `physics_code` 参数，通过 `physics_proj` 注入 action 分支

### （2）InverseHead —「给定未来 latent，反推动作」

```
输入: z_t [B,48], z_tH [B,48], text_embed
输出: pred_action [B,10,7]
Loss: L_inv = MSE(pred_action, action_chunk)
```

- **代码**：`models/world2wam_heads.py` → `class InverseHead`

### （3）ActionAdapter —「直接从当前 latent 预测动作」（推理用）

```
输入: z_t [B,48], text_embed
输出: pred_action [B,10,7]
Loss: L_act（见下文，MLP/LightDiT 为 MSE，FlowDiT 为 flow loss）
```

- **MLP 版**：`class ActionAdapter`
- **LightDiT 版**：`class LightActionDiT`（Transformer + learnable action query tokens）
- **FlowDiT 版**：`models/action_dit.py` → `class FlowActionDiT`

## 4.2 Cycle Consistency（循环一致性）

```
z_pred = ForwardHead(z_t, action_chunk, text)
a_cycle = InverseHead(z_t, z_pred, text)
L_cycle = MSE(a_cycle, action_chunk)
```

**直觉**：Forward 再 Inverse 应能还原原动作，保证世界模型与逆模型一致。

## 4.3 总损失函数

```
L_total = λ_act·L_act + λ_fwd·L_fwd + λ_inv·L_inv + λ_cycle·L_cycle
```

默认权重（paper 配置）：λ_act=1, λ_fwd=1, λ_inv=1, λ_cycle=0.1

**代码**：`train/training_utils.py` → `compute_world2wam_losses()`

## 4.4 两阶段训练脚本

| 阶段 | 脚本 | 训练什么 |
|------|------|----------|
| Stage1 | `train/train_world2wam_heads.py` | 仅 Forward + Inverse |
| Stage2 | `train/train_world2wam_adapter.py` | Forward + Inverse + Adapter |

Stage2 支持：
- `--warm_start_heads`：从 Stage1 checkpoint 加载
- `--adapter_type mlp|light_dit|flow_dit`
- `--max_samples`, `--max_steps`：debug 用

## 4.5 已有实验结果（idea2）

| 实验 | Checkpoint | 结果 |
|------|------------|------|
| Stage1 | `experiments/world2wam_heads_paper/heads_final.pt` | val loss ≈ **0.038** |
| Stage2 MLP | `experiments/world2wam_adapter_paper/adapter_final.pt` | train loss ≈ **0.112** |
| Offline eval 50k | `experiments/eval_offline_paper_300k.json` | mse_fwd=**0.037**, mse_act=**4.63** |

**解读**：ForwardHead 泛化很好；动作预测 MSE 仍较大，adapter 有提升空间。

---

# 五、FlowActionDiT：Flow Matching 动作专家（新增，对齐 FastWAM）

## 5.1 为什么需要 FlowActionDiT？

FastWAM 官方动作头是 **diffusion / flow matching**，不是单步回归。  
`LightActionDiT` 虽有 Transformer 结构，但训练仍是 **MSE 回归**，与 FastWAM 推理路径不一致。

**FlowActionDiT** 实现与 FastWAM 同族的 **velocity field 学习 + ODE 采样**。

## 5.2 数学原理（老师版）

### 训练时

设真实动作为 `a0`，噪声为 `ε`，随机时间 `τ ∈ [0,1]`：

```
a_τ = (1 - τ)·a0 + τ·ε          # 在真实动作和噪声之间插值
v_target = ε - a0                 # 目标速度场
模型预测: v_θ = FlowActionDiT(z_t, text, a_τ, τ)
Loss: L_flow = MSE(v_θ, v_target)
```

### 推理时

从纯噪声 `x ~ N(0,I)` 出发（τ=1），沿 velocity 场积分到 τ=0：

```
for τ from 1 down to 0:
    v = model(z_t, text, x, τ)
    x = x - dt * v
return x   # 即为预测 action chunk
```

默认 `num_steps=10`（Euler 积分）。

## 5.3 网络结构

```
cond_tokens = [state_token, text_tokens, time_token, (physics_token)]
action_tokens = Linear(noisy_action)
tokens = concat(cond_tokens, action_tokens)
→ TransformerEncoder (batch_first, norm_first, GELU)
→ 取最后 H 个 action tokens → Linear → velocity [B,10,7]
```

- **文件**：`models/action_dit.py`
- **类**：`FlowActionDiT`
- **方法**：`forward()`, `compute_flow_loss()`, `sample()`

## 5.4 与 idea2 训练集成

- `adapter_type=flow_dit` 时，`compute_world2wam_losses` 走 flow loss 分支
- Forward/Inverse/Cycle **仍用真实 action_chunk**（稳定）
- 日志含：`loss_action_flow`, `flow_tau_mean`, `mse_act_sample`（低频采样评估）

## 5.5 配置与命令

- 配置：`configs/world2wam_libero_spatial_h10_flow_dit.yaml`
- 单元测试：`tests/test_flow_action_dit.py`（✅ 已通过）

---

# 六、LightActionDiT：轻量 DiT 单步回归（C 方案）

## 6.1 结构

```
z_t + text (+ physics_code) → cond vector
learnable action_queries [H, D] + cond → Transformer → action [B,H,7]
```

- **文件**：`models/world2wam_heads.py` → `class LightActionDiT`
- **参数量**：约 8.5M
- **训练**：单步 MSE，非 diffusion

## 6.2 状态

| 项 | 状态 |
|----|------|
| 代码集成 | ✅ |
| Forward shape 测试 | ✅ |
| 正式 300k 训练 | ⚠️ CPU debug 曾中断，GPU 正式训练待跑 |

---

# 七、idea3 Physics v1：物理阶段对齐（最新完成）

## 7.1 动机

v0 问题：physics 伪标签 **塌缩**到 `push_slide` + `transport` 两类，Router 学不到有效阶段。

**v1 目标**：motion/contact phase-aware conditioning（非完整 ProPhy）。

## 7.2 八个 Phase 类别

```
free_motion   — 空移，场景变化小
approach      — 接近目标，尚未交互
contact       — 开始接触
grasp         — 夹爪闭合抓取
transport     — 携带物体移动
place         — 放置/释放
push_slide    — 推动/滑动物体
uncertain     — 规则冲突或置信度低
```

## 7.3 PhysicsPhaseLabeler v1（伪标签生成）

**原理**：从 action 运动学 + latent 变化 +（可选）state/gripper + 关键词规则，输出：

```
phase_id [B]        — 0~7
confidence [B]      — [0.05, 1.0]，用于加权训练
```

**特征示例**：
- `motion_mag`：action xyz 位移幅度
- `latent_delta`：||z_tH - z_t||
- `gripper_delta`：夹爪开合变化
- `horizontal_motion` / `vertical_motion`

**代码**：
- `physics/phase_labeler.py` → `class PhysicsPhaseLabeler`
- `physics/physics_labels.py` → v0/v1 路由，`batch_infer_physics_labels_v1()`

**检查工具**：`tools/inspect_physics_labels.py`（输出分布、confidence、warnings）

## 7.4 PhysicsPhaseRouter（阶段路由器）

**训练时输入**：
```
concat(z_t, z_tH-z_t, text, action_features, state_t, state_tH)
```

**推理时输入**（不能用未来 z_tH）：
```
concat(z_t, zeros, text, zeros, state_t, zeros)  # 缺失特征用零填充
```

**输出**：
```
phase_logits [B,8] → softmax → phase_prob [B,8]
physics_code [B, physics_dim] = Linear(phase_prob)
```

**代码**：`models/physics_router.py` → `class PhysicsPhaseRouter`, `class PhysicsConditioner`

## 7.5 PhysicsAlignedWorld2WAM 包装器

把 Router 的 `physics_code` 注入整条链路：

```
physics_code → FiLM 调制 z_t（PhysicsConditioner）
            → ForwardHead / InverseHead（可选 physics_proj）
            → ActionAdapter / LightDiT / FlowDiT（internal 或 external residual）
```

**代码**：`models/physics_world2wam.py` → `class PhysicsAlignedWorld2WAM`

## 7.6 Physics v1 损失函数

```
L = idea2_loss
  + λ_phase · confidence_weighted_CE(router_logits, pseudo_phase)
  + λ_balance · (-entropy(batch_mean_phase_prob))   # 防止 Router 塌缩
  + λ_phy_delta · cosine_direction_loss(z_pred_H - z_t, z_tH - z_t)
  + λ_phy_smooth · action 时序平滑
  + λ_phy_gripper · gripper 维加权 MSE
```

**代码**：`train/physics_losses.py` → `compute_physics_losses()`

## 7.7 Physics 训练脚本

- **文件**：`train/train_physics_world2wam.py`（已重写为 v1）
- **配置**：
  - `configs/world2wam_libero_spatial_h10_physics_v1.yaml`（MLP）
  - `configs/world2wam_libero_spatial_h10_physics_flow_dit_v1.yaml`（FlowDiT）
- **支持**：`--adapter_type flow_dit`, `--phase_label_version v1`, `--use_physics true` 等

## 7.8 Physics v1 实验状态

| 项 | 状态 |
|----|------|
| 单元测试 `tests/test_physics_v1.py` | ✅ 7 项全过 |
| CPU 3-step MLP physics 训练 | ✅ |
| CPU 2-step FlowDiT physics 训练 | ✅ |
| Label inspect 5000 样本 | ⚠️ 无 task text 时 uncertain 偏高，需更大样本或接入任务描述 |
| GPU 正式 20k step 训练 | ❌ 待 GPU |
| LIBERO physics sim eval | ❌ 待 GPU |

---

# 八、评估体系

## 8.1 Offline Cache Eval（不加载 FastWAM，算 MSE）

**文件**：`eval/eval_offline_cache_only.py`

```
mse_fwd, mse_inv, mse_cycle, mse_act / mse_act_sample
```

Physics 模式额外输出：`phase_acc_pseudo`, `phase_entropy`, `phase_counts_eval`

## 8.2 LIBERO Sim Eval（需要 GPU + 仿真）

**文件**：`eval/eval_libero_world2wam.py`

| Mode | 含义 |
|------|------|
| `baseline` | 官方 FastWAM |
| `ours_dit` | FastWAM 官方多步 diffusion |
| `ours_onestep_mlp` | 我们的 MLP 单步 |
| `ours_onestep_light_dit` | LightDiT 单步 |
| `ours_onestep_flow_dit` | FlowDiT ODE 采样 |
| `ours_residual_*` | a = a_fastwam + α·ours_action |
| `ours_onestep_physics_*` | Physics Router 推理 + adapter |
| `ours_residual_physics_*` | Physics + residual |

---

# 九、完整代码改动清单（按模块）

## 9.1 新增文件

| 文件 | 作用 |
|------|------|
| `models/action_dit.py` | FlowActionDiT + flow matching |
| `physics/phase_labeler.py` | Physics v1 伪标签 + confidence |
| `tools/inspect_physics_labels.py` | 标签分布检查 + 验收 warnings |
| `eval/physics_eval_utils.py` | Physics 模型加载 helper |
| `tests/test_flow_action_dit.py` | FlowDiT 单元测试 |
| `tests/test_physics_v1.py` | Physics v1 单元测试 |
| `configs/world2wam_libero_spatial_h10_flow_dit.yaml` | FlowDiT Stage2 配置 |
| `configs/world2wam_libero_spatial_h10_physics_v1.yaml` | Physics MLP 配置 |
| `configs/world2wam_libero_spatial_h10_physics_flow_dit_v1.yaml` | Physics FlowDiT 配置 |

## 9.2 重大修改文件

| 文件 | 改动摘要 |
|------|----------|
| `models/world2wam_heads.py` | LightActionDiT；`build_action_adapter`；Forward/Inverse/MLP 加 `physics_code`；flow_dit 分支 |
| `models/physics_router.py` | v1：z_tH-z_t、train/infer 双模式、零填充、输出别名 |
| `models/physics_world2wam.py` | v1：inject 开关、`forward_inference`、FlowDiT 支持 |
| `physics/physics_labels.py` | v0/v1 路由、`batch_infer_physics_labels_v1` |
| `train/training_utils.py` | flow loss 分支；`sample_action_adapter`；physics checkpoint |
| `train/train_world2wam_adapter.py` | flow_dit CLI + 日志 |
| `train/train_physics_world2wam.py` | **完全重写**为 Physics v1 |
| `train/physics_losses.py` | confidence CE、entropy balance、class balance |
| `data/latent_cache_dataset.py` | 加载 state_t/state_tH；`detect_state_dim` |
| `eval/eval_offline_cache_only.py` | physics ablation、`mse_act_sample` |
| `eval/eval_libero_world2wam.py` | flow_dit modes + 6 个 physics modes |

## 9.3 adapter_type 兼容机制

```python
# models/world2wam_heads.py
resolve_adapter_type()  # mlp | light_dit | flow_dit
build_action_adapter()  # 按类型构造模型

# checkpoint 含 adapter_type 字段；旧 ckpt 无此字段 → 默认 mlp
```

---

# 十、实验资产索引

| 资产 | 路径 |
|------|------|
| Latent cache 300k | `cache/libero_spatial_h10_full_fastwam/` |
| Stage1 heads | `experiments/world2wam_heads_paper/heads_final.pt` |
| Stage2 MLP adapter | `experiments/world2wam_adapter_paper/adapter_final.pt` |
| Offline eval 50k | `experiments/eval_offline_paper_300k.json` |
| idea3 v0 debug | `experiments/world2wam_idea3_physics_debug/` |
| Physics v1 CPU debug | `experiments/world2wam_physics_v1_debug/` |
| FlowDiT physics debug | `experiments/world2wam_physics_flow_dit_debug/` |

---

# 十一、当前不足与下一步（诚实汇报）

## 11.1 未完成 / 待验证

1. **FlowDiT / LightDiT GPU 正式训练**（300k cache，20k steps）
2. **Paper 级 LIBERO sim eval**（baseline vs ours_* vs physics_*）
3. **Physics 伪标签质量**：cache 无原始 task 文本，keyword 规则无效，uncertain 占比可能偏高
4. **600k cache 扩展**（resume 脚本有小 bug）
5. **Stage2 warm-start 正式重训**（代码已支持）

## 11.2 与 FastWAM 官方 Action DiT 的差距

| 维度 | 我们 v1 | FastWAM 官方 |
|------|---------|--------------|
| 条件注入 | concat tokens + Transformer | AdaLN-Zero + MoT mask |
| 训练 | action-only flow | video-action 联合 flow |
| Latent | 48-d pooled | token-level video latent |
| 物理 | rule-based phase | ProPhy REB |

## 11.3 建议下一步（给老师看的研究路线）

1. GPU 上跑 FlowDiT + Physics v1 正式训练 → offline MSE 对比 MLP baseline
2. 接入 LIBERO task 文本改善 phase 伪标签
3. Sim eval：residual_flow_dit + physics_residual 是否优于 FastWAM baseline
4. 长期：unpooled latent → token-level physics routing（MoT）

---

# 十二、可复制运行命令（附录）

```bash
export PY=/DATA/disk0/jianhua/miniconda3/envs/world2wam/bin/python
cd /DATA/disk0/jianhua

# 1. 检查 physics 标签分布
$PY "Physics-Aligned World2WAM/tools/inspect_physics_labels.py" \
  --cache_dir cache/libero_spatial_h10_full_fastwam \
  --max_samples 5000 --phase_label_version v1 \
  --output_json experiments/physics_label_inspect_v1.json

# 2. Physics v1 CPU debug 训练
$PY "Physics-Aligned World2WAM/train/train_physics_world2wam.py" \
  --config configs/world2wam_libero_spatial_h10_physics_v1.yaml \
  --cache_dir cache/libero_spatial_h10_full_fastwam \
  --warm_start_heads experiments/world2wam_heads_paper/heads_final.pt \
  --adapter_type mlp --max_samples 500 --max_steps 20 --device cpu

# 3. FlowDiT Stage2 训练
$PY "Physics-Aligned World2WAM/train/train_world2wam_adapter.py" \
  --config configs/world2wam_libero_spatial_h10_flow_dit.yaml \
  --adapter_type flow_dit --warm_start_heads experiments/world2wam_heads_paper/heads_final.pt \
  --max_steps 20000 --device cuda

# 4. 单元测试
$PY "Physics-Aligned World2WAM/tests/test_physics_v1.py"
$PY "Physics-Aligned World2WAM/tests/test_flow_action_dit.py"
```

---

# 十三、飞书文档建议目录（给 GPT 直接用）

```
1. 汇报摘要（3 句话）
2. 研究背景：机器人操作 + FastWAM + 我们的切入点
3. 总体方案架构图
4. 数据：300k Latent Cache 是什么
5. idea2：Forward / Inverse / Adapter + Cycle Loss
6. Action Adapter 三代演进：MLP → LightDiT → FlowActionDiT
   6.1 Flow Matching 原理（公式 + 直觉）
   6.2 FlowActionDiT 网络结构
7. idea3 Physics v1：8 phase + Router + physics_code 注入
   7.1 伪标签 v1 规则
   7.2 Router 训练/推理差异
   7.3 损失函数与防塌缩
8. 训练与评估流程
9. 实验结果表格
10. 代码仓库结构与关键文件
11. 当前局限与下一步计划
12. 附录：命令行与 checkpoint 路径
```

---

*End of report for GPT → Feishu conversion.*

# Flow Corrector × World2WAM Version D：下一步实现与实验指南

## 1. 文档目的

这份文档给 Flow Corrector 分支的开发同学直接开工使用。目标不是把两套方法简单拼接，而是在不破坏 World2WAM 主线、公平协议和可复现性的前提下，回答三个问题：

1. Flow Corrector 单独加在 FastWAM 上是否有效？
2. World2WAM Version D 单独是否有效？
3. Flow Corrector 能否在 Version D 之上继续带来互补增益？

论文中的方法主次暂定为：

- **主方法：World2WAM Version D。** 训练期使用 Forward / Inverse / Cycle 未来动力学约束，并通过 action-priority gradient projection 处理辅助目标与动作目标的梯度冲突；推理时删除辅助头，不执行 future rollout。
- **可选增强：Flow Corrector。** 在冻结策略输出的粗动作上运行一个轻量条件流场，使用少量迭代进行动作细化。它允许增加少量推理开销，因此必须单独报告准确率—延迟权衡。

## 2. 当前方法接口

### 2.1 Version D 的推理接口

当前统一入口位于：

```text
version_d/runtime/policy_lora/src/wrappers/fastwam_wrapper.py
```

主要接口：

```python
out = wrapper.forward_action_only(batch)
a0 = out["pred_action"]
```

现状：`forward_action_only` 只返回 `pred_action`，没有返回 Flow Corrector 所需的条件表征 `c`。

训练阶段的 wrapper 已经通过 MoT forward hook 捕获 action tokens，并在 `extract_hidden` 中实现了 mask-aware pooling。因此不要重新设计一套不一致的视觉特征提取器，应复用同一种 action-relevant hidden 定义。

### 2.2 Version D 的 checkpoint

Version D 的 LoRA 在导出时被合并进 FastWAM action expert，辅助 future / inverse heads 不进入部署 checkpoint：

```text
version_d/runtime/policy_lora/src/tools/export_libero_checkpoint.py
```

Flow Corrector 应把导出的 Version D checkpoint 当作冻结 base policy。不要让 Flow loss 回传进 Version D，也不要在第一轮实验中联合训练两者。

### 2.3 Flow Corrector 的目标形式

冻结 base policy 后：

```math
a_0 = \pi_{\mathrm{base}}(o), \qquad a_1 = a_{\mathrm{GT}}
```

采用 rectified-flow 线性路径：

```math
a_t = (1-t)a_0 + t a_1, \qquad u_t = a_1-a_0
```

训练条件速度场：

```math
\mathcal L_{\mathrm{flow}}
= \mathbb E_{t\sim U(0,1)}
\left\|v_\theta(a_t,t,c)-(a_1-a_0)\right\|_2^2.
```

推理时从 `a0` 出发做 Euler 更新：

```math
a_{k+1}=a_k+\Delta t\,v_\theta(a_k,t_k,c).
```

## 3. 推荐的无侵入结合方式

整体数据流：

```text
observation + language + proprio
                │
                ▼
 frozen FastWAM / frozen exported Version D
                │
        ┌───────┴────────┐
        ▼                ▼
   coarse action a0   condition c
        └───────┬────────┘
                ▼
         Flow Corrector
                │
                ▼
      corrected action a_star
```

第一阶段必须采用**串行、冻结、后训练**方案：

1. 完成并导出 base FastWAM 或 Version D。
2. 冻结整个 base policy。
3. 离线生成 `(condition, a0, a_gt, metadata)` 缓存。
4. 只训练 Flow Corrector。
5. 评测时同时保留 `a0` 与 `a_star`，便于 paired failure analysis。

这样做的优点：

- 不改变 Version D 的训练目标和主结论。
- Flow Corrector 的增益可被独立归因。
- 可以预计算大模型输出，小头训练不必反复运行 5B backbone。
- Flow 失败时可以回退到原始动作。

## 4. 条件表征 c 的具体选择

### 4.1 首选方案

使用 **action expert 最后一层、动作输出头之前的 pooled hidden**：

- 它同时包含图像、语言、proprio 和动作生成相关信息。
- 它与 Version D 的 LoRA 更新处于同一 action expert 语义空间。
- 现有 `extract_hidden` 已经提供 mask-aware pooling，可减少重复实现。

建议接口：

```python
out = wrapper.forward_action_only(batch, return_features=True)
a0 = out["pred_action"]
c = out["action_condition"]
```

约束：

- `c` 必须 `detach()`。
- 保存 `feature_source`、`feature_dim`、pooling 方式和 normalization 配置。
- base 和 Version D 必须使用同一特征位置和同一归一化方式。
- 不建议仅用 raw visual encoder feature，因为它可能缺少语言、proprio 和动作生成状态。

### 4.2 接口实现建议

优先顺序：

1. 给 wrapper 增加 `return_features=False` 可选参数。
2. 在一次 `infer_action` 中通过稳定 hook 捕获 action tokens。
3. 使用已有 `extract_hidden` 或抽取公共 pooling helper。
4. 默认路径保持原返回值与原速度，不影响没有 Flow 的评测。

不要依赖私有临时变量、层编号硬编码或重复执行一次 backbone forward 来取特征。

### 4.3 必做一致性测试

开启与关闭 `return_features` 时：

```text
max_abs(pred_action_plain - pred_action_feature_mode) < 1e-6
```

若 BF16 环境存在非确定误差，需要记录并给出合理阈值，但两条路径不能产生系统性动作变化。

## 5. 正确的训练目标

### 5.1 FastWAM + Flow

```text
source = official FastWAM action a0_fastwam
target = expert action a_gt
condition = FastWAM action hidden c_fastwam
```

### 5.2 Version D + Flow

```text
source = exported Version D action a0_version_d
target = expert action a_gt
condition = Version D action hidden c_version_d
```

必须为 Version D 重新生成 source action 和 condition。不要把只在 FastWAM 残差分布上训练的 Corrector 直接套到 Version D，因为 Version D 的误差幅度、方向和失败任务分布都可能变化。

### 5.3 不推荐的 teacher 方案

不建议令：

```text
target = Version D output
```

这会把 Flow Corrector 变成 Version D 蒸馏器，原则上只能逼近 teacher，很难证明二者结合后继续提升。只有在“把 Version D 能力蒸馏给更便宜 base”这一独立目标下才使用 teacher target。

## 6. 缓存格式与数据红线

建议每个样本保存：

```json
{
  "benchmark": "libero",
  "suite": "libero_spatial",
  "task_id": 0,
  "episode_id": 0,
  "sample_id": 0,
  "base_method": "VersionD_B5_s42",
  "base_checkpoint_sha256": "...",
  "feature_source": "action_expert_final_pooled",
  "feature_dim": 1024,
  "action_shape": [8, 7],
  "train_seed": 42
}
```

Tensor 数据至少包括：

```text
condition
pred_action_init
expert_action
action_is_pad
```

数据红线：

- LIBERO-Plus 是 OOD 评测集，不能使用其七种测试扰动 rollout 来训练 Flow Corrector。
- 训练只使用标准 LIBERO 官方训练数据。
- FastWAM + Flow 和 Version D + Flow 使用相同的训练样本清单、动作归一化和训练步数。
- 不根据 Plus 结果更换任务、seed 或挑选 checkpoint。

## 7. 模型和优化建议

### 7.1 第一版模型

建议先保留现有轻量配置：

```text
hidden_size: 512
num_layers: 3
action_horizon: 8
action_dim: 7
sigma_noise: 0
```

训练建议：

```text
optimizer: AdamW
learning_rate candidates: 1e-4, 3e-4
weight_decay: 1e-4
precision: BF16
gradient_clip_norm: 1.0
EMA: optional, fixed before formal runs
```

先用固定小规模 pilot 决定单一学习率，正式三种子时不得再根据结果改超参数。

### 7.2 防止修坏正确动作

LIBERO 标准任务成功率已经接近饱和，Corrector 最大风险是修改本来正确的动作。建议增加 gated residual：

```math
a^* = a_0 + \alpha(a_0,c)\cdot(a_{\mathrm{flow}}-a_0),
\qquad \alpha\in[0,1].
```

实现要求：

- gate 初始值偏向 0，训练初期接近 identity。
- 输出经过官方 action bounds clip。
- 记录每个 episode 的 `alpha` 均值、P95 和修正范数。
- 支持显式 `disable_flow` 回退。

可以加入 correction-norm regularization：

```math
\mathcal L = \mathcal L_{\mathrm{flow}}
+ \lambda_{\Delta}\|a^*-a_0\|_2^2,
```

但 `lambda_delta` 必须在 pilot 阶段冻结，不能用正式测试结果反向调参。

### 7.3 动作平滑与安全

必须处理：

- action clipping；
- gripper 维度的离散/连续语义；
- first-order action delta；
- second-order jerk；
- NaN/Inf 自动回退 `a0`；
- correction norm 超阈值时回退或缩放。

## 8. Flow 是否真的必要：关键强基线

当前直线路径的目标速度恒为 `a_gt - a0`。在确定性 source/target 和单一专家动作下，审稿人可能认为它等价于复杂化的残差回归。

因此必须增加同参数量级的单步残差基线：

```math
a^* = a_0 + g_\phi(a_0,c).
```

至少比较：

| Corrector | 推理调用次数 | 目的 |
|---|---:|---|
| Identity | 0 | 原策略 |
| Residual MLP | 1 | 检查单步残差是否足够 |
| Flow | 1 | 检查 flow 训练目标本身 |
| Flow | 2 | 低开销迭代 |
| Flow | 4 | 推荐候选 |
| Flow | 8 | 饱和趋势 |
| Flow | 10 | 当前默认 |

只有多步 Flow 在相近参数量和训练数据下明显优于 residual MLP，才适合把 Flow Matching 作为主要技术贡献。否则应诚实称为 lightweight action corrector，并采用最有效、最便宜的版本。

## 9. 冻结实验矩阵

### 9.1 核心四组

| ID | Base policy | Corrector | 作用 |
|---|---|---|---|
| F0 | FastWAM | 无 | 官方基线 |
| F1 | FastWAM | Flow-F | Flow 单独贡献 |
| D0 | Version D | 无 | World2WAM 主方法 |
| D1 | Version D | Flow-D | 组合与互补性 |

其中 Flow-F 与 Flow-D：

- 架构、参数量和优化器完全相同；
- 训练数据 sample IDs 完全相同；
- 分别从各自 base policy 生成 `a0` 和 `c`；
- target 都是同一份 `a_gt`。

### 9.2 必要消融

1. Residual MLP vs Flow。
2. `c`：无条件 / 视觉 pooled / action hidden pooled。
3. Euler steps：1 / 2 / 4 / 8 / 10。
4. gate：关闭 / 开启。
5. correction norm regularization：关闭 / 冻结后的单一权重。
6. base：FastWAM / Version D。

不要把所有组合都跑成笛卡尔积。先用固定 pilot 选接口和步数，再冻结正式协议。

## 10. Benchmark 顺序

### 10.1 第一阶段：离线与 smoke test

- CPU shape test。
- checkpoint load/save test。
- base output consistency test。
- 100–500个标准 LIBERO validation samples 上比较 action MSE。
- 10个固定 simulator episodes 检查动作范围、NaN 和接口。

### 10.2 第二阶段：LIBERO-Plus 15% 固定子集

使用与 World2WAM 主流水线相同的 task manifest、seed=42 和初始状态：

- FastWAM；
- FastWAM + Flow；
- Version D；
- Version D + Flow；
- Residual MLP 强基线。

必须按 Camera、Robot Init、Language、Light、Background、Noise、Layout 分项报告。

### 10.3 第三阶段：标准 LIBERO 与完整 Plus

只有 smoke test 没有协议错误后才运行完整表：

- 标准 LIBERO：确认 Corrector 没有破坏 clean performance。
- LIBERO-Plus：验证 OOD robustness。
- 三个训练种子：42 / 43 / 44。

### 10.4 RoboTwin

LIBERO 组合确认有效后，再移植到 RoboTwin。不要在两个 benchmark 同时调接口，否则无法区分算法问题和环境问题。

## 11. 指标与统计

除成功率外，必须报告：

- paired success delta；
- 95% paired bootstrap CI；
- McNemar test；
- 相对错误率下降；
- steps-to-success；
- inference latency mean / P50 / P95；
- peak GPU memory；
- corrector 参数量和 checkpoint 大小；
- action first-order change；
- jerk；
- correction norm；
- gate alpha；
- base 成功但 corrector 失败的 regression count；
- base 失败但 corrector 成功的 recovery count。

核心诊断表：

| 类型 | 定义 |
|---|---|
| preserved success | base 成功，corrector 成功 |
| recovered failure | base 失败，corrector 成功 |
| introduced regression | base 成功，corrector 失败 |
| persistent failure | base 失败，corrector 失败 |

组合方法要成立，不能只看 recovered failure；introduced regression 必须足够低。

## 12. 代码任务清单

建议在独立分支完成以下工作，避免直接修改正式流水线。

### P0：接口与测试

- [ ] 为 `FastWAMWrapper.forward_action_only` 增加可选 feature 返回。
- [ ] 抽取公共 action-token pooling helper。
- [ ] 增加 feature/no-feature 输出一致性测试。
- [ ] 完成 Flow head CPU smoke test。
- [ ] 完成 action shape、mask、dtype、device 测试。
- [ ] 对 NaN、过大 correction 和 checkpoint 不匹配实现安全回退。

### P1：离线缓存与训练

- [ ] 实现 `precompute_flow_cache.py`。
- [ ] 缓存中写入 base checkpoint SHA256 和 sample manifest SHA256。
- [ ] 实现 FastWAM cache 与 Version D cache 两套 manifest。
- [ ] 训练时拒绝 checkpoint hash 不匹配的 cache。
- [ ] 每500步保存可恢复 checkpoint。
- [ ] 输出完整 resolved config 和训练 seed。

### P2：评测

- [ ] 把 Flow backend 接入统一 LIBERO evaluator。
- [ ] 支持 `identity / residual / flow` 三种模式。
- [ ] 支持 `num_steps` 配置。
- [ ] 输出每 episode 的 `a0`/`a_star` 统计，不保存超大动作 tensor。
- [ ] 接入 latency、VRAM、smoothness 和 failure-type 记录。
- [ ] 复用 World2WAM 固定 task/episode manifest。

### P3：论文产物

- [ ] 四组核心表。
- [ ] residual-vs-flow 表。
- [ ] Euler step—success—latency 曲线。
- [ ] 七种扰动分项图。
- [ ] recovered/regressed failure 配对统计图。

## 13. 验收门槛

工程验收：

- 原 FastWAM / Version D 不开启 Flow 时输出不变。
- cache 与 checkpoint hash 可追踪。
- 训练可从最近 checkpoint 恢复。
- 评测中异常动作自动回退，不导致整条任务退出。
- 一条汇总命令可重新生成 CSV 和 LaTeX 表。

实验验收：

- 所有方法使用同一 episode manifest。
- 不挑选最好 seed。
- 标准 LIBERO 不出现不可接受的成功率回退。
- LIBERO-Plus 至少在若干预先定义扰动上产生稳定 paired gain。
- 组合 D1 的增益必须高于 D0，同时说明增加的延迟。
- 若 Flow 不优于 residual MLP，则降低其论文地位，不强行包装。

## 14. 推荐的第一周执行顺序

1. 接通 `action_condition`，完成输出一致性测试。
2. 用一个固定 Version D checkpoint 生成500样本 cache。
3. 跑 residual MLP 与 1/4-step Flow pilot。
4. 做10个固定 simulator episode，排除动作范围和 gripper 问题。
5. 冻结模型结构、学习率、步数和 gate 配置。
6. 生成 FastWAM 与 Version D 的同样本全量 cache。
7. 训练 Flow-F / Flow-D 三种子。
8. 进入 LIBERO-Plus 15% 固定子集配对评测。

## 15. 最终论文表述建议

如果组合有效：

> World2WAM improves the policy representation through training-time bidirectional future-dynamics constraints and action-priority gradient projection. A lightweight conditional flow corrector can be optionally attached to the exported policy to further refine residual action errors under distribution shifts.

如果 Flow 只在少数扰动有效：

> The corrector is presented as an optional robustness–latency trade-off rather than part of the core World2WAM inference path.

如果 Flow 与 residual MLP 相当：

> 保留动作校正思想，但采用更简洁的 residual corrector；不要声称多步 Flow 是必要贡献。

核心原则：Version D 的主结论不能依赖 Flow；Flow 的价值必须通过同协议、同任务实例、强残差基线和推理代价共同证明。


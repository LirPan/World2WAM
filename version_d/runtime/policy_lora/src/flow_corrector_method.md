# Flow Matching 动作校正头（Flow Corrector）—— 我的方法说明

> 给学长的评审稿。对应三实验设计里的「① 只有你的方法（base + Flow，无 LoRA）」和「③ 结合」里我的那一半。
> 零侵入：只新增文件，不碰学长的 `train_lora_*` / `fastwam_wrapper.py` / `inverse_action_head.py`。

---

## 0. TL;DR（一句话）

在**冻结的 base FastWAM** 推理输出 `a0` 之上，架一个**条件 Flow Matching 速度场** `vθ(a_t, t, c)`，用 Euler 积分把 base 的"粗动作"沿条件流场细化为 `a*`，作为最终动作。不重训 base、不引入 LoRA，只是对 base 输出做一次**有监督、条件化**的校正。

对应导师提的 "Flow Matching (VLA-Corrector / OGG) 接入 action decoder" 的思路，落点在 FastWAM 的动作头上。

---

## 1. 动机

- base FastWAM 在推理时已经删掉未来视频生成分支、直接出动作（`forward_action_only`）。它给的 `a0` 是"单步直接回归"的动作，**没有显式的迭代细化**。
- 动作生成可以看成从「base 预测」到「专家动作」的一条**传输路径**；用 Flow Matching 学这条路径的速度场，能在**不重训 base、不引入 LoRA** 的前提下，对 base 输出做一次条件化的校正。
- 这是导师提的 Flow Matching 校正思路在 FastWAM 动作头上的具体落地。

---

## 2. 方法（形式化）

记动作维度 `D = 7`（LIBERO），动作跨度 `T = 8`（action_horizon），条件维度 `C`（默认 1024，来自 base 的观测表征）。

### 2.1 传输路径（rectified-flow / 条件流匹配）

- 源：`a0 = base FastWAM.forward_action_only(o)`（**冻结，不回传**）。
- 目标：`a1 = a_gt`（数据集里的专家动作），训练时的监督目标。
- 线性插值路径：`a_t = (1 - t) · a0 + t · a1`，`t ∈ [0, 1]`。
- 该路径的速度：`u_t = da_t/dt = a1 - a0`。

### 2.2 速度场网络 `vθ(a_t, t, c)`

- 输入：`a_t ∈ R^{B×T×D}`、流时间 `t ∈ [0, 1]`（正弦时间嵌入到 `time_embed_dim`）、条件 `c ∈ R^{B×C}`。
- 结构：`FlowMatchingActionHead`（MLP / Transformer 栈，`num_layers=3`，`hidden_size=512`），输出与输入同形 `R^{B×T×D}` 的速度。
- `c` 来源（待接线，见 §4）：目前 batch 里透传 `flow_cond`；最终接 base FastWAM 视觉编码器输出的**冻结**表征。

### 2.3 训练目标 `flow_loss`

```
t      ~ U(0, 1)
a_t    = (1 - t) · a0 + t · a1
target = a1 - a0
L      = E[ || vθ(a_t, t, c) - target ||^2 ]
```

标准 conditional flow matching / rectified flow 回归，**只训练 `vθ` 参数**，base 全程冻结。

### 2.4 推理（校正）

```
a0 = base.forward_action_only(o)          # 粗动作
a* = Euler 积分 vθ, 从 a0 出发, num_steps=10, σ_noise=0（确定性）
       a_{k+1} = a_k + Δt · vθ(a_k, t_k, c)
返回 a*
```

`FlowCorrector.correct(wrapper, batch)` 就是这条流水线：调 `forward_action_only` 拿 `a0`，再 `head.sample(c, x_source=a0)` 得 `a*`，返回
`{ pred_action: a*, pred_action_init: a0, used_flow: True }`。

---

## 3. 为什么是"校正"而不是"重生成"

- 起点是 base 的**真实输出** `a0`，不是纯噪声；速度场学的是 `a1 - a0`（**残差式修正**），所以推理是在 base 之上做小幅、有方向的条件化微调，而不是从噪声采样一个新动作。
- 这保证两点：
  1. 不脱离 base 的语义分布；
  2. 即使正确头没充分训练，`a0` 仍是可用的回退动作（graceful degradation）。

---

## 4. 与 base FastWAM 的接口（零侵入）

- 接入口：学长 `runtime/policy_lora/src/models/`（action_expert_adapter / inverse_action_head / future_latent_head）+ `wrappers/action_only_inference.py` 的 `forward_action_only`。
- `FlowCorrector` **只只读消费** `forward_action_only` 的 `pred_action` 输出，不修改学长任何 `train_lora_*` 脚本 / 配置。
- 新增文件（全在 `version_d/runtime/policy_lora/src/` 下）：
  - `models/flow_matching_action_head.py` —— 速度场网络 + `flow_loss` + `sample`
  - `wrappers/flow_corrector.py` —— 包 `forward_action_only` → `head.sample`
  - `smoke_test_flow_action.py` —— CPU 无卡冒烟（5 项校验，用假 wrapper）
  - `eval/eval_flow_action.py` —— LIBERO 评测薄封装（三实验开关 scaffold）
  - `flow_README.md` —— 内部速查

---

## 5. 实现状态

| 项 | 状态 |
|---|---|
| 速度场头 + corrector + smoke test | ✅ 已写，提交在 `wenjie/flow-action`（parent=senior-main `3a681e9`，diff 恰好 5 文件、0 修改学长代码） |
| CPU 无卡 smoke test | ✅ 已就绪（等 torch 装完即跑，不加载权重） |
| `eval_flow_action.py` | 🚧 scaffold：完整 LIBERO 评测需 (1) 学长同款 **conditioned** FastWAM ckpt；(2) LIBERO 仿真环境；(3) 一张空闲 GPU（当前 NY / FA 被占，三实验暂无法跑） |
| 条件 `c` 接线 | 🚧 目前 batch 透传 `flow_cond` 占位，最终接 base 编码器冻结输出——下一步接线重点 |

---

## 6. 三实验设计与对比口径（红线）

- **① 只有你的方法** = base（冻结）+ Flow corrector（无 LoRA）
- **② 只有我的方法** = base + LoRA / version D（无 Flow）→ 跑学长已有 pipeline 取 standalone 数
- **③ 结合并行** = base + LoRA + Flow corrector

**对比红线**：必须用学长同款 **conditioned** FastWAM ckpt + 官方 `eval_libero_single`，**不要**用我之前那个 uncond 权重跑出的 95.8% 当基线——两者协议 / 权重不同，不可比。

---

## 7. 想跟学长确认的几个点（开放问题）

1. `c` 用 base 编码器的哪一段输出最合适（最后一层 hidden / pool / 某中间层）？是否允许我 freeze 读它。
2. 训练时 `a1` 取 GT 动作是否合适，还是想让 `a1` 取"学长 LoRA 版输出"做 teacher（影响 ③ 怎么联合训）。
3. 推理期 `num_steps`（Euler 步数）和 `σ_noise` 有没有经验值；先确定性（`σ=0`）跑 baseline 是否 OK。
4. 训练数据：直接用 LIBERO 的 `(obs, action)` 对，还是复用学长 pipeline 里的同一批 episode 以保证可比。

---

*分支：`wenjie/flow-action`；parent：`senior-main` (`3a681e9`)。本 MD 与 5 个新增文件一并作为"你的方法"交付物。*

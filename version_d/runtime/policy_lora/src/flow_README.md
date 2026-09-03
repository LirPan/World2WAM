# Flow Matching Action Corrector（你的独立贡献）

> 分支 `wenjie/flow-action`（基于 `senior-main` = 学长 main @ 3a681e9）。零侵入：只新增文件，不改学长 `train_lora_*` / `fastwam_wrapper.py` / `inverse_action_head.py`。

## 三个实验（与学长 2026-09-03 确认）
统一冻结 base FastWAM，叠两个开关 `use_lora`（学长 LoRA）、`use_flow_corrector`（你的 Flow 头）：

| 实验 | use_lora | use_flow_corrector | 对应 |
|------|:---:|:---:|------|
| ① 只有你的方法 | ✗ | ✓ | base + Flow corrector |
| ② 只有我的方法 | ✓ | ✗ | base + LoRA（跑学长 pipeline 取数） |
| ③ 结合并行 | ✓ | ✓ | base + LoRA + Flow corrector |

评测入口：`eval/eval_flow_action.py --use-lora/--no-use-lora --use-flow-corrector`。

## 新增文件
- `models/flow_matching_action_head.py` — 条件速度场 `vθ(a_t,t,cond)` + Euler 积分 `sample()` + `flow_loss()`（2.1）
- `wrappers/flow_corrector.py` — 包裹冻结 FastWAM，`forward_action_only` 出 a0 后 flow 细化（2.2）
- `eval/eval_flow_action.py` — 三实验开关薄封装
- `smoke_test_flow_action.py` — 无卡冒烟（CPU 即可，不加载 FastWAM 权重）

## 运行顺序（等腾卡）
1. ② 先用学长 pipeline 在 libero-plus 取 LoRA standalone 基线（不动文件）
2. ① 实现+跑 Flow 单独：`python eval/eval_flow_action.py --no-use-lora --use-flow-corrector`
3. ③ 结合：`--use-lora --use-flow-corrector`

## 红线
- 必须由学长同款 **conditioned** FastWAM ckpt + 官方 `eval_libero_single`；**不可用你的 uncond 95.8%** 对比。
- cond 推荐复用 FastWAM vision tower 输出（`batch["flow_cond"]`），不重训视觉编码器。
- OGG 不可直迁，需你设计适配层（本项目即在做这件事）。

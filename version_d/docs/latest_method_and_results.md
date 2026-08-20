# Version D：最新方法与结果说明

## 一句话概括

Version D 没有重做整个 FastWAM，而是在官方 FastWAM 权重上增加一个轻量 LoRA 适配层，并用“未来状态表征预测 + 正向/逆向/循环一致性”约束训练；同时用梯度冲突投影保护主任务动作学习。

## 具体改变

### 1. 主干保持不变

R0 直接使用官方 `robotwin_uncond_3cam_384.pt`，作为冻结的 baseline。这样比较时，变化主要来自新增训练目标与 LoRA 参数，而不是更换模型、数据或评测器。

### 2. 用 LoRA 做参数高效适配

LoRA 加在 attention 的 `q/k/v/o` 投影上，rank=8、alpha=16。训练时主要更新 LoRA 及新增的预测头，避免全量改写大模型；导出时将适配结果合并成评测器可以直接加载的 checkpoint。

### 3. 增加未来 latent 预测

给定当前观测和语言任务表示，模型不仅预测当前动作，还预测下一时刻的 latent 表征。训练前会缓存目标 future latent，减少训练阶段反复计算 teacher 表征的开销。

### 4. 增加 FIC 一致性约束

当前使用四项损失：

```text
L = 1.00 L_action
  + 0.10 L_forward
  + 0.05 L_inverse
  + 0.05 L_cycle
```

- `L_action`：动作模仿，始终是主目标。
- `L_forward`：当前表征预测未来 latent。
- `L_inverse`：由目标未来 latent 反推当前表征，避免表征只会单向漂移。
- `L_cycle`：当前 → 未来 → 当前 的闭环一致性。

### 5. 处理多目标梯度冲突

辅助 world-model/FIC 目标有时会把梯度推向与动作目标相反的方向。Version D 在合并梯度前执行 `project_conflicts`：当辅助梯度与动作梯度冲突时，去掉冲突方向；不冲突的方向仍然保留。因此它的作用不是“强行提高每个任务”，而是降低辅助目标破坏动作控制的风险。

### 6. 提高困难任务采样比例

训练中约 70% 样本来自困难任务关键词：`dual`、`three`、`stapler`、`hammer`、`cabinet`、`switch`、`stamp`。这使训练信号更多落在 baseline 容易失败的任务上，但也意味着必须用固定任务集和 paired evaluation 检查是否出现泛化损失。

## 实验流程

服务器上的流程是：准备官方 checkpoint 和数据统计 → 准备 ActionDiT/文本 embedding → 分片缓存 future latent → 训练 LoRA+FIC → 导出合并 checkpoint → 用同一个 RoboTwin evaluator 评测 R0 与 R3 → 从 `summary.json` 汇总 clean/random 结果。

R3 当前配置为：3000 steps、batch size 1、learning rate `1e-4`、bf16、future-loss warmup 500 steps、随机种子 42。R0 不经过 R3 训练，直接用官方 checkpoint 评测。

## 最新 paired validation

四组数据各包含 5 个任务，每个方法每个任务 10 个 episode，其中 5 个 clean、5 个 random：

| 任务组 | R0 clean | R3 clean | R0 random | R3 random | R0 平均 | R3 平均 |
|---|---:|---:|---:|---:|---:|---:|
| fixed5 | 72% | 72% | 60% | 64% | 66% | 68% |
| next5B | 68% | 68% | 72% | 70% | 70% | 69% |
| next5C | 66% | 70% | 64% | 66% | 65% | 68% |
| next5D | 64% | 66% | 62% | 56% | 63% | 61% |
| **四组平均** | **67.5%** | **69.0%** | **64.5%** | **64.0%** | **66.0%** | **66.5%** |

所以，当前能严谨说的是：R3 相比 R0 在 clean 上提高 1.5 个百分点，总平均提高 0.5 个百分点；random 平均下降 0.5 个百分点。这个结果说明方法有正向信号，但还不能表述为“已经显著超过 baseline”，因为目前仍是小规模 validation，不是完整 benchmark。

此前还有一轮 3-episode clean-only smoke：R0 为 73.3%，R3 为 66.7%。样本太少，而且没有 random 条件，因此只用于发现问题，不与上面的 n=10 paired validation 混合。

## 下一步

1. 完成同一任务全集上的 full benchmark，并固定 episode seeds。
2. 保存每组 R0/R3 的原始 `summary.json`、CSV 和运行返回码。
3. 分别报告 clean、random、hard-task 子集，避免只报一个平均数。
4. 若 random 下降继续存在，优先检查困难任务过采样比例、LoRA scale 和 FIC loss 权重，再做单变量 ablation。


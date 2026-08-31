# World2WAM 当前成果汇报（2026-08-31）

## 一分钟结论

我们已经完成 World2WAM Version D 的核心方法、RoboTwin 标准配对验证，以及 LIBERO 的训练和 checkpoint 导出。当前最可信的正向结果是：在 RoboTwin 固定 hard-10 任务上，Version D 相比官方 FastWAM，在 clean 条件下从 47% 提升到 54%（+7 个百分点），在 random 条件下从 42% 提升到 45%（+3 个百分点）。两组结果都覆盖10个任务、每任务10次，并且不存在缺失任务。

因此，目前可以确认方法存在正向效果，并且 random 条件下没有出现此前小规模实验中的整体退化。不过 hard-10 是预先固定的困难任务子集，不是完整 RoboTwin benchmark；LIBERO 的 Version D 成功率仍在等待仿真评测，现阶段不能宣称全面超过 FastWAM。

## 已经完成了什么

1. 在官方 FastWAM 动作专家的 q/k/v/o attention projection 中加入 rank-8、alpha-16 LoRA。
2. 保留动作 imitation loss 作为主目标，并增加 Forward、Inverse、Cycle 三个训练期约束。
3. 实现动作优先的梯度冲突投影：辅助 world loss 与动作 loss 冲突时，只移除辅助梯度的冲突分量，不修改动作梯度。
4. 导出时将 LoRA 合并回 FastWAM，Forward/Inverse/Cycle heads 不进入部署模型，因此推理路径不增加 future rollout。
5. 完成 RoboTwin hard-10 的 clean/random 标准配对评测。
6. 完成 LIBERO Spatial 的12,000条 future-latent缓存、6,000-step训练和合并 checkpoint 导出。
7. 在另一台服务器独立完成 LIBERO 1,000样本、500-step pilot 的训练和导出，用于复核链路。

## 当前标准结果

| Benchmark / 条件 | FastWAM R0 | Version D | 提升 | 完整性 |
|---|---:|---:|---:|---|
| RoboTwin hard-10 clean | 47% | 54% | +7 pp | 10/10任务完成 |
| RoboTwin hard-10 random | 42% | 45% | +3 pp | 10/10任务完成 |
| LIBERO-90 FastWAM baseline | 15.13% | 待评测 | — | baseline 90任务、4500 episodes完成 |
| LIBERO Spatial matched | 待同协议复评 | checkpoint已导出 | — | 仿真评测排队中 |

RoboTwin 协议为官方 standard evaluator、关闭 GraphLite、相同任务和 episode 数。结果源文件位于服务器：

`/DATA/disk0/yjh/robotwin_w2wam/runs/robotwin_hard10_standard_pair_n10_v2/summary.json`

## 方法为什么可能有效

FastWAM 推理时直接输出动作，速度快，但只靠动作监督可能无法充分保留“执行动作后场景会怎样变化”的结构。Version D 在训练阶段强制当前表征同时满足三件事：能够预测下一时刻 latent、能够从未来 latent 恢复动作、预测结果再反推动作时保持一致。

辅助目标并不总是有利于动作控制。训练日志中多次观察到 action gradient 与 world gradient 的余弦为负。Version D 在这种情况下投影掉辅助梯度的冲突方向，使动作目标保持最高优先级。最终只导出适配后的 FastWAM 权重，因此这是一种“训练时增强世界理解、推理时保持快速动作路径”的方案。

## 现在还缺什么

1. LIBERO Spatial official/Version D 的同任务、同初始状态配对成功率。
2. RoboTwin 更广的固定20任务评测，证明提升不只存在于 hard-10。
3. Action-only LoRA、naive F/I/C、F/I/C+projection 三组关键消融。
4. 三个训练种子的均值和标准差。
5. 推理延迟、峰值显存、可训练参数量、训练GPU-hours、95%置信区间。

## 论文故事线

论文问题可以表述为：如何在不恢复昂贵 test-time future imagination 的情况下，让 FastWAM 获得更强的未来感知表征和随机扰动鲁棒性？

核心回答是：使用训练期双向 latent dynamics 提供 future-aware regularization，再以动作优先的冲突投影避免辅助任务破坏控制学习；推理时删除辅助 heads，保留 FastWAM 的低延迟优势。

当前数据已经足以开始写 Method、Introduction、Related Work、Experimental Setup 和 RoboTwin 初版结果。最终 performance claim 要等待 LIBERO、扩大任务覆盖和核心消融完成后冻结。

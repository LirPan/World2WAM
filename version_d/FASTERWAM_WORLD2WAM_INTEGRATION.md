# FasterWAM + World2WAM

## 结论

Version D 可以迁移到 FasterWAM。FasterWAM 作为更强的稀疏 World Action Model 骨干，World2WAM 只在训练期增加 forward future-latent、inverse action 和 cycle consistency 监督，并在 action-priority 模式下投影冲突梯度。推理时删除辅助 head，不做 future rollout，保留 FasterWAM 原生 sparse MoT 推理路径。

这构成公平的四路比较：

1. FastWAM；
2. FastWAM + World2WAM（Version D）；
3. FasterWAM；
4. FasterWAM + World2WAM。

不能把 FasterWAM 的收益直接归因于 World2WAM；论文主张应由同一骨干内的 paired comparison 和消融支持。

## 已完成的兼容性修复

- 新增 `configs/fasterwam_world2wam_robotwin.yaml`，固定 FasterWAM checkpoint、统计文件、14 维 action、1024 hidden size 和 Version D 的 F/I/C + conflict projection 配置。
- 新增 `configs/robotwin_fasterwam_world2wam_3cam_384_1e-4.yaml`，使用 FasterWAM 的 SparseMoT task 配置；官方完整 checkpoint 已包含 action expert，因此不再依赖不存在的 standalone SparseActionDiT checkpoint。
- 新增 `scripts/run_fasterwam_world2wam_train.sh`，训练后将 LoRA 合并并导出为 FasterWAM 原生 `.pt`，辅助 head 不进入推理 checkpoint。
- 新增 `scripts/queue_fasterwam_world2wam_auto.sh`，按空闲 GPU 顺序运行 seed 42/43/44，带 GPU 二次确认、单 seed 锁和失败停机保护。
- RoboTwin evaluator 改为优先复用已存在的共享 model cache，避免把空目录误传给 ModelScope，修复 `ModuleNotFoundError: modelscope`。
- 训练脚本把文本缓存映射到 policy 工作目录和 FasterWAM 工作目录，修复相对路径导致的 `Missing text embedding cache`。

## 当前验证

- FasterWAM + Version D adapter construction smoke test 已通过：action dim=14、hidden dim=1024、LoRA trainable parameters=3,538,944。
- FastWAM/Version D 的 RoboTwin 单任务 smoke test 已通过，clean/random 均可完成。
- FasterWAM seed42 正式训练已重新启动；导出前必须同时存在 `.pt` 和 `.pt.sha256`。
- RoboTwin FastWAM baseline 与 Version D seed42 主表正在重新评测；首轮失败原因已定位为错误 model-cache 路径，不是算法失败。

## 运行方式

```bash
# 单个 seed
/DATA/disk0/yjh/world2wam_iclr2027/deploy/run_fasterwam_world2wam_train.sh 42 1

# 三个 seed 自动接续
nohup /DATA/disk0/yjh/world2wam_iclr2027/deploy/queue_fasterwam_world2wam_auto.sh \
  > /DATA/disk0/yjh/world2wam_iclr2027/runs/paper_sprint_v2/logs/fasterwam_world2wam_auto.out 2>&1 &
```

## 论文解释边界

标准 LIBERO 上 FasterWAM baseline 已达到约 99.1%，高于当前 FastWAM+Version D 的约 97.05%，所以不能预先声称迁移后一定全面 SOTA。正确的实验问题是：在更强骨干上，World2WAM 是否仍能改善困难任务、RoboTwin 和 LIBERO-Plus 的分布外扰动鲁棒性，同时保持推理成本不增加。

最终表格必须使用同一任务、初始状态、episode seed、重规划设置，并报告均值、标准差、paired CI、最差任务和推理延迟。任何旧结果若缺少 manifest/checkpoint hash，只能标为 exploratory。

## 待补缺口

- FasterWAM + World2WAM 三个 seed 的训练与导出；
- FasterWAM baseline 和 Version D 在 RoboTwin 50-task clean/random 主表；
- LIBERO-Plus 的 matched FastWAM/FasterWAM/Version D 对照；
- F/I/C、aligned projection、hard sampling 的因果消融；
- 延迟、显存、参数量、训练 GPU-hours、梯度冲突比例和失败类型统计；
- 统一汇总命令、manifest 完整性检查和最终 paired bootstrap 表格。

# World2WAM Version D：离线续工与最快出结果手册

更新时间：2026-08-27（Asia/Shanghai）

这份文档的目标是：即使暂时连不上两台服务器，也能直接继续写论文、审计结果、修改代码和准备下一轮实验；服务器恢复后，按固定顺序在最短时间内补齐可信结果。

## 0. 先记住当前结论

当前最可信的正向信号是：

- RoboTwin hard-10，标准官方评测、关闭GraphLite、每任务10次；
- Fast-WAM R0 clean成功率为47%；
- Version D R3 clean成功率为54%；
- 提升7个百分点。

目前不能声称“Version D全面超过Fast-WAM”，原因是：

- hard-10 random有3个模拟器任务失败；
- random原始聚合的两种方法分母不同；
- 在双方都有结果的8个random任务上，R0为47.5%，Version D为45.0%；
- LIBERO只完成Fast-WAM baseline，Version D尚未完成训练和配对评测。

论文可以立即开始写Method、Related Work、Experimental Setup和结果表框架；最终性能claim必须等待random补跑、关键消融和LIBERO配对结果。

## 1. 仓库快照与入口

本次同步包含：

- `version_d/runtime/policy_lora/src/`：两台服务器实际运行的Version D源码快照；
- `version_d/configs/`：RoboTwin 14D和LIBERO 7D配置；
- `version_d/scripts/`：标准配对、hard-10并行评测、LIBERO流水线；
- `version_d/eval/`：当前可审计结果摘要；
- `version_d/tools/analyze_paired_results.py`：强制使用相同任务集合重新计算R0/Version D比较。

核心实现：

- `runtime/policy_lora/src/train/train_lora_fic_hardtask.py`
- `runtime/policy_lora/src/losses/world2wam_losses.py`
- `runtime/policy_lora/src/models/future_latent_head.py`
- `runtime/policy_lora/src/models/inverse_action_head.py`
- `runtime/policy_lora/src/wrappers/fastwam_wrapper.py`

R4行为锚定分支保存在
`runtime/policy_lora/src/train/train_lora_fic_r4.py`，但当前只有n=2探索结果，不是论文默认方法。

## 2. 已完成工作的准确状态

### 2.1 方法实现

R3 / Version D已经实现：

1. 在Fast-WAM attention的q/k/v/o投影中插入LoRA，rank=8、alpha=16；
2. 保留原始Fast-WAM动作损失作为主目标；
3. Forward Head：由当前隐藏表征与真实动作预测下一时刻latent；
4. Inverse Head：由当前隐藏表征与目标future latent反推动作；
5. Cycle：由预测future latent再次恢复动作；
6. 对共享LoRA参数执行action-prioritized conflict projection；
7. 只投影辅助world梯度，完整保留动作梯度；
8. future/inverse分支只在训练时使用，导出后不增加未来rollout；
9. RoboTwin困难任务采样比例为70%；
10. 训练日志记录梯度余弦、冲突标记与两类梯度范数。

总目标：

```text
L_action weight = 1.00
L_forward weight = 0.10
L_inverse weight = 0.05
L_cycle weight = 0.05
```

辅助损失在前500步warm-up。RoboTwin配置为14D动作；LIBERO配置为7D动作，两者checkpoint不能混用。

### 2.2 RoboTwin结果

#### A. hard-10标准n=10：当前主结果

原始文件：
`eval/20260825_robotwin_hard10_standard_pair_n10.json`

| 条件 | R0 | Version D | 差值 | 是否完整 |
|---|---:|---:|---:|---|
| clean，10个任务 | 47.0% | 54.0% | +7.0 pp | 是 |
| random，双方完整的8个任务 | 47.5% | 45.0% | -2.5 pp | 否 |

失败的random项：

- R0：beat_block_hammer
- Version D：beat_block_hammer
- Version D：press_stapler

原始JSON中的random聚合分别基于9个和8个任务，不能直接比较。

#### B. fixed-5标准n=3：只作诊断

原始文件：
`eval/20260825_robotwin_fixed5_standard_pair_n3.json`

| 条件 | R0 | Version D | 差值 |
|---|---:|---:|---:|
| clean | 60.0% | 80.0% | +20.0 pp |
| random | 86.7% | 73.3% | -13.4 pp |

每任务仅3次，不能写成显著提升或SOTA。

#### C. R4探索结果

原始文件：
`eval/20260817_robotwin_r4_hard10_n2_exploratory.json`

R4 clean 45%、random 30%，每任务只有2次。它只证明训练、导出和评测链路可运行，不支持论文结论。

#### D. 应排除的旧结果

FiveAges上旧fixed-5 n=10中，R0关闭GraphLite而R3开启GraphLite，协议不一致，且R3明显下降。这组结果只能用于说明“评测协议曾经有问题”，不能进入主表，也不能与标准结果合并。

### 2.3 LIBERO结果

原始摘要：
`eval/20260825_libero90_fastwam_baseline_summary.json`

- 官方Fast-WAM checkpoint：`libero_uncond_2cam224.pt`
- 90个任务
- 每任务50次
- 共4500 episodes
- 成功681次
- 成功率15.13%

旧的`_libero90_final_agg.txt`指向过期目录，错误显示0/0。本仓库数字是直接遍历90个结果JSON重新计算得到。

注意：该LIBERO-90结果不是当前Version D的matched LIBERO-Spatial实验，也不能与论文中不同协议的四套LIBERO数字直接比较。

Version D当前只完成：

- LIBERO环境与官方Fast-WAM baseline链路；
- Spatial训练数据确认；
- 10条任务文本embedding预计算。

尚未完成：

- future latent完整预计算；
- LIBERO 7D Version D训练；
- checkpoint导出；
- official/Version D相同任务、seed、trials的配对评测。

## 3. 明天离线时可以直接做的工作

### 3.1 第一个15分钟

```bash
cd /path/to/World2WAM
git pull origin main

python version_d/tools/analyze_paired_results.py \
  version_d/eval/20260825_robotwin_hard10_standard_pair_n10.json

python version_d/tools/analyze_paired_results.py \
  version_d/eval/20260825_robotwin_fixed5_standard_pair_n3.json

python -m compileall -q version_d/runtime version_d/scripts version_d/tools
```

预期hard-10输出：

- clean matched_task_count=10，delta=+7.0 pp；
- random matched_task_count=8，delta=-2.5 pp；
- random complete=false。

### 3.2 论文写作优先级

不需要服务器即可完成：

1. Method：写LoRA、F/I/C、冲突投影与训练/推理差异；
2. Related Work：Fast-WAM、latent world models、多任务梯度冲突；
3. Experimental Setup：统一任务、seed、成功判定、GraphLite关闭；
4. 主表模板：RoboTwin clean/random/hard、LIBERO Spatial；
5. 消融表模板：Action-only、F、F+I、F+I+C、F+I+C+projection；
6. 效率表模板：训练参数量、GPU-hours、推理延迟、显存；
7. Failure Analysis：random下降、模拟器失败、困难任务过采样的潜在偏置。

论文方法标题建议：

> Conflict-Aware Bidirectional Latent Dynamics Adaptation for Efficient World-Action Models

当前可写的贡献：

1. 双向latent动力学约束，使未来状态可预测且动作可恢复；
2. 动作优先的非对称梯度投影，降低辅助目标的负迁移；
3. 训练期使用辅助分支，推理保持Fast-WAM路径。

在消融完成前，不要写“projection已被证明是提升来源”；只能写成待验证的设计。

### 3.3 离线代码工作

优先完成以下审计：

- 为`_aligned_backward`补单元测试：
  - 正内积不投影；
  - 负内积投影后与动作梯度内积约为0；
  - 动作梯度不被修改；
  - 无world梯度/无action梯度时行为正确。
- 为F/I/C损失补shape、dtype和mask测试；
- 为hard-10汇总补“分母不同则拒绝比较”测试；
- 检查LoRA导出后Future/Inverse Head不会进入推理；
- 固定所有实验的任务列表、seed和输出命名。

## 4. 服务器恢复后的第一优先级

### 4.1 先补hard-10 random失败项

不要重跑已经成功的37个method-task-phase组合。当前runner会检查结果文件，只补缺失项。

把最新runner同步到New_yjh后执行：

```bash
/DATA/disk0/yjh/robotwin_w2wam/env/bin/python \
  /DATA/disk0/yjh/robotwin_w2wam/run_robotwin_hard10_parallel_new_yjh.py \
  --episodes 10 \
  --gpus 1,2,3,4,5,6 \
  --output-root /DATA/disk0/yjh/robotwin_w2wam/runs/robotwin_hard10_standard_pair_n10_v2
```

完成条件：

- `failures=[]`
- clean和random均`matched_task_count=10`
- 两个方法所有任务返回码为0
- 原始结果文件、日志和summary均保留

如果完整random仍下降：

1. 不扩大评测；
2. 先做单变量消融；
3. 将hard sampling从0.7降至0.5；
4. 再测试world loss scale或LoRA scale；
5. 不要同时改多个变量。

### 4.2 跑LIBERO Version D

New_yjh配置已经修正为实际路径：
`configs/libero_version_d_new_yjh.yaml`

一键脚本：
`scripts/run_libero_version_d_new_yjh.sh`

正式运行前必须确认：

```bash
test -f /DATA/disk0/yjh/libero_work_wj/checkpoints/fastwam_release/libero_uncond_2cam224.pt
test -f /DATA/disk0/yjh/libero_work_wj/checkpoints/fastwam_release/dataset_stats.json
test -d /DATA/disk0/yjh/libero_work_wj/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot
test -d /DATA/disk0/yjh/world2wam/plr/yjh_space_backup_20250602/idea2_workspace/code/LIBERO_fresh
```

先用500-step pilot配置做3任务代码烟测，目的只是在几十个rollout内发现环境、导出或路径错误：

```bash
LIBERO_VERSION_D_CONFIG=/DATA/disk0/yjh/robotwin_w2wam/latest/code/policy_lora/configs/libero_version_d_new_yjh_pilot.yaml \
LIBERO_VERSION_D_RUN=/DATA/disk0/yjh/libero_work_wj/runs/libero_version_d_spatial_pilot \
LIBERO_CACHE_MAX_SAMPLES=1000 \
LIBERO_TASK_IDS="0 1 2" \
LIBERO_NUM_TRIALS=10 \
LIBERO_TRAIN_GPU=0 \
LIBERO_EVAL_GPU=1 \
bash run_libero_version_d_new_yjh.sh
```

烟测成功后，使用固定10任务做matched评测。论文最终表建议至少50次/任务；10次只用于选择是否继续。

```bash
LIBERO_TASK_IDS="0 1 2 3 4 5 6 7 8 9" \
LIBERO_NUM_TRIALS=50 \
LIBERO_TRAIN_GPU=0 \
LIBERO_EVAL_GPU=1 \
bash run_libero_version_d_new_yjh.sh
```

Go条件建议：

- 相同10任务、seed、trials和evaluator；
- Version D整体至少提升3个百分点，或大多数任务同方向提升；
- 不允许通过删除失败任务改变分母；
- 推理延迟与官方Fast-WAM基本一致；
- 至少有Action-only、FIC-naive、FIC+projection三组消融。

若LIBERO无提升：

1. 先确认7D动作、dataset stats和future latent来自同一数据版本；
2. 检查训练日志中的gradient conflict比例和余弦；
3. 比较FIC-naive与projection；
4. 将hard sampling设为0，排除采样偏置；
5. 再调整world loss，不要先更换benchmark。

## 5. 跨benchmark迁移判断

Version D可迁移的最低数据接口是：

```text
current observation + language + action + next observation + success evaluator
```

需要实现五个适配点：

1. dataset adapter输出当前/下一时刻观测与动作；
2. Fast-WAM或等价主干产生当前hidden和target future latent；
3. 配置正确的action_dim与动作归一化；
4. 复用F/I/C和conflict projection；
5. 建立同checkpoint起点、同任务和同seed的baseline/Version D评测。

### 推荐顺序

#### 第一：LIBERO

已经有官方checkpoint、数据、评测器和7D配置，是唯一适合“马上出结果”的第二benchmark。不要在LIBERO闭环前分散到第三套环境。

#### 第二：CALVIN，小规模迁移验证

官方仓库：[mees/calvin](https://github.com/mees/calvin)

优点：

- 语言条件；
- 7D连续笛卡尔动作与当前LIBERO接口接近；
- 长时序任务很适合验证latent dynamics故事；
- 官方评测允许通过`CustomModel.step(obs, goal)`接入新策略。

成本：

- 需要CALVIN dataset adapter和独立主干/checkpoint；
- 官方环境依赖较旧；
- 长时序正式评测耗时较高。

建议只在LIBERO出现正向结果后，先用debug数据和single-step设置验证可迁移性。

#### 第三：ManiSkill

官方仓库：[mani-skill/ManiSkill](https://github.com/mani-skill/ManiSkill)

GPU并行仿真速度快，适合未来扩大任务，但当前缺少Fast-WAM兼容数据与语言接口，迁移成本高于LIBERO和CALVIN。

#### 暂缓：RoboCasa365

官方网站：[RoboCasa](https://robocasa.ai/)

RoboCasa365规模大、任务长且科学价值高，但包含大量厨房任务、场景和演示，适合作为后续扩展，不适合当前ICRA冲刺阶段从零接入。

#### 备选：RLBench

官方仓库：[RLBench](https://github.com/stepjam/RLBench)

任务丰富且支持语言描述，但CoppeliaSim环境、数据格式和评测适配工作量较大。只有团队中有人已经有可用环境时才考虑。

## 6. 实验决策树

```text
补齐RoboTwin random
        |
        +-- hard clean正、random不降 --> 冻结RoboTwin主表
        |
        +-- random继续下降 ---------> hard sampling / loss scale单变量消融

完成LIBERO Version D
        |
        +-- matched提升 >= 3 pp ----> 写双benchmark主线，开始CALVIN小迁移
        |
        +-- 接近持平 --------------> 强调hard-task与机制，补projection消融
        |
        +-- 明显下降 --------------> 暂停第三benchmark，先修7D适配与训练冲突
```

## 7. 论文最小可投稿证据

必须完成：

- RoboTwin hard-10 clean/random完整matched结果；
- LIBERO至少一个标准suite的matched结果；
- Action-only、F、F+I+C、F+I+C+projection；
- 梯度余弦/冲突率曲线；
- 推理延迟与参数量；
- 失败分析；
- 所有结果保留原始JSON、任务、seed、返回码。

最好完成：

- 三个训练seed或bootstrap置信区间；
- hard sampling消融；
- world loss scale消融；
- CALVIN小规模可迁移性结果。

不能做：

- 更换或削弱baseline协议；
- 删除失败episode后重新计算；
- 把n=2/n=3写成显著结果；
- 把LIBERO-90的15.13%与不同协议的论文数字直接比较；
- 把未完成的后台任务写成结果。

## 8. 当前最短路径

1. 离线完成Method与实验表格框架；
2. 补hard-10 random三个失败项；
3. 同时启动LIBERO future latent预计算；
4. 先跑LIBERO三任务烟测，再跑10任务matched n=10；
5. 正向后扩大到n=50；
6. 只做能解释结果的关键消融；
7. LIBERO闭环后再考虑CALVIN，不要先开第三条大工程。

## 9. 2026-08-27 晚间恢复运行记录

- New_yjh 已部署 `wait_and_run_priority_new_yjh.sh`：检测到真正空闲 GPU 后，先补齐 RoboTwin hard10 的3个缺失随机化任务，再执行LIBERO-Spatial Version D小规模pilot。两个阶段均支持断点续跑。
- New_yjh 队列默认每10秒扫描全部8张卡；FiveAges 的 Version D 队列也默认每10秒扫描全部8张卡，只在显存不超过1GB且GPU利用率不超过5%时接管空闲卡。它不会抢占或终止别人的活动进程。
- FiveAges 的LIBERO90评测发现PyTorch 2.6+兼容问题：`torch.load`默认启用`weights_only=True`，导致可信的LIBERO初始状态文件无法反序列化。两台服务器均已将该调用改为显式`weights_only=False`。
- 可复用补丁位于 `version_d/patches/libero_torch26_weights_only.patch`。该参数只应对项目自带、来源可信的LIBERO init-state文件使用。

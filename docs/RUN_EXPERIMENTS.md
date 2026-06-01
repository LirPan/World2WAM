# 实验运行指南（World2WAM Minimal）

## 0. 后台跑大任务（关 Cursor 也不断）

在服务器上执行，使用 `nohup` 脱离终端：

```bash
cd /DATA/disk1/yjh_space/idea2_workspace/minimal_world2wam

# 一键：下载资源 + Pipeline B（02→03→04）
bash scripts/bg_launch.sh full_pipeline

# 只看下载
bash scripts/bg_launch.sh download_assets

# 查看状态 / 跟日志
bash scripts/bg_launch.sh status
bash scripts/bg_launch.sh tail full_pipeline
```

- 日志：`experiments/bg_jobs/full_pipeline.log`
- PID：`experiments/bg_jobs/full_pipeline.pid`

## 1. 创建 Conda 环境

```bash
cd /DATA/disk1/yjh_space/idea2_workspace/minimal_world2wam
bash scripts/setup_conda_env.sh
conda activate world2wam
```

重装环境：

```bash
RECREATE=1 bash scripts/setup_conda_env.sh
```

## 2. 前置资产检查清单

| 资产 | 路径 / 命令 | 用途 |
|------|-------------|------|
| Wan2.2-TI2V-5B | `FastWAM/checkpoints/` + `DIFFSYNTH_MODEL_BASE_PATH` | 模型加载、VAE |
| ActionDiT backbone | `FastWAM/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt` | `preprocess_action_dit_backbone.py` |
| LIBERO LeRobot 数据 | `FastWAM/data/libero_mujoco3.3.2/*_lerobot/` | 训练 / 预计算 |
| T5 text cache（全量 FastWAM 训练） | `precompute_text_embeds.py` | 原仓库 baseline 训练 |
| **官方 FastWAM checkpoint（必需）** | `FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt` | baseline / World2WAM 共用，配置项 `official_fastwam_checkpoint` |
| Future latent cache | `data/future_latents/world2wam_minimal/*.pt` | 训练前 `02_precompute` |

数据下载（FastWAM README）：

```bash
cd /DATA/disk1/yjh_space/idea2_workspace/code/FastWAM
mkdir -p data/libero_mujoco3.3.2
# 从 https://huggingface.co/datasets/yuanty/LIBERO-fastwam 下载 tar.gz 并解压
```

模型准备：

```bash
cd /DATA/disk1/yjh_space/idea2_workspace/code/FastWAM
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/fastwam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
# 按 FastWAM README 下载 Wan 权重到 checkpoints/
```

## 3. 框架分层测试

```bash
conda activate world2wam
cd /DATA/disk1/yjh_space/idea2_workspace/minimal_world2wam
bash scripts/smoke_test_framework.sh
```

- **Tier 0**：不依赖 FastWAM 权重（head / loss / cache）
- **Tier 1**：`import fastwam`
- **Tier 2**：LeRobot 数据集（需数据目录）
- **Tier 3**：加载完整 `FastWAMWrapper`（需 checkpoints）

## 4. 实验流水线（minimal_world2wam）

### 4.1 预计算 future latent

```bash
conda activate world2wam
cd minimal_world2wam
bash scripts/02_precompute_future_latents.sh
# 调试: --max-samples 100
```

输出：`data/future_latents/world2wam_minimal/*.pt`

若数据不在默认路径，编辑 `configs/fastwam_future_distill.yaml`：

```yaml
lerobot_dataset_dirs:
  - /abs/path/to/libero_spatial_no_noops_lerobot
  # ...
```

### 4.2 World2WAM-Probe（frozen，只训 FutureLatentHead）

```bash
bash scripts/03_train_future_distill.sh
# 或显式: --backbone-mode frozen
```

损失：`L_total = L_action + current_lambda_fwd * L_future`（frozen 时 L_action 不更新 backbone）。

### 4.3 World2WAM-Policy（LoRA / adapter，追求 success）

```bash
bash scripts/07_train_world2wam_policy_improve.sh
# smoke: bash scripts/07_train_world2wam_policy_improve.sh --max-steps 20
```

产物：`experiments/world2wam_policy_improve/checkpoints/world2wam_final.pt`、`resolved_config.yaml`、`logs/*.json`

### 4.4 Bidirectional 表征分析（非 success 主路径）

```bash
bash scripts/05_train_bidirectional_world2wam_smoke.sh   # cycle smoke
bash scripts/06_eval_bidirectional_heads.sh             # 离线 head MSE
```

默认 `backbone_mode=frozen`，仅训练 auxiliary heads；**不保证** LIBERO success 提升。

### 4.5 Action-only 评估

```bash
bash scripts/04_eval_action_only.sh --max-batches 5
# 官方 baseline（默认 official_fastwam_checkpoint）:
bash scripts/04_eval_action_only.sh
# Policy 训练后:
bash scripts/04_eval_action_only.sh --checkpoint experiments/world2wam_policy_improve/checkpoints/world2wam_final.pt
```

`eval_results.json` 含 `future_head_called: false`、`uses_future_video: false`、`checkpoint`。

### 4.6 FastWAM baseline（可选，一般**不需要**重训）

仅当需要从头训练 FastWAM 时：

```bash
bash scripts/01_run_fastwam_baseline.sh
```

论文对照实验请直接用 **官方 checkpoint**，见 4.5 / 步骤 A。

### 4.7 LIBERO 仿真成功率（原仓库 eval）

```bash
conda activate world2wam
cd ../code/FastWAM
python experiments/libero/eval_libero_single.py ckpt=/path/to/checkpoint
```

## 5. 推荐实验顺序（论文主线）

| 步骤 | 内容 | 命令 |
|------|------|------|
| **A** | FastWAM 官方 checkpoint baseline eval（不重训 FastWAM） | `bash scripts/04_eval_action_only.sh` |
| **B** | World2WAM-Probe：`backbone_mode=frozen` | `bash scripts/03_train_future_distill.sh` |
| **C** | World2WAM-Policy：`backbone_mode=lora` | `bash scripts/07_train_world2wam_policy_improve.sh` |
| **D** | Bidirectional 表征分析（smoke + 离线 MSE） | `05` smoke + `06` eval |
| **E** | Action-only：官方 vs `world2wam_final.pt` | `04` 两次对比 |
| **F** | LIBERO sim success | `bash scripts/run_libero_spatial_success.sh` |
| **F'** | 官方 vs World2WAM sim 对比 | `bash scripts/run_compare_libero_success.sh` |

**强调**：

- 不需要重训 FastWAM baseline；统一 `official_fastwam_checkpoint`。
- Probe（B）与 Bidirectional（D）只分析 hidden / world↔action 映射，**不保证** success 提升。
- Policy（C）才是追求 LIBERO success 的主实验。
- 推理始终 **action-only**（不加载 FutureLatentHead / InverseActionHead / future video）。

前置：`02_precompute_future_latents.sh` → `data/future_latents/world2wam_minimal/`

## 6. 常见问题

| 现象 | 处理 |
|------|------|
| `future_latent is missing` | 先跑 `02_precompute_future_latents.sh` |
| `Repository path does not exist` | 检查 config 中 `fastwam_root` |
| `dataset` Tier 2 SKIP | 下载 LeRobot LIBERO 到 `FastWAM/data/...` |
| Tier 3 SKIP | 准备 Wan + ActionDiT checkpoint |
| CUDA OOM | 减小 `batch_size`；precompute 用 `--max-samples` |

## 7. 环境变量

```bash
export WORLD2WAM_CONDA_ENV=world2wam
export DIFFSYNTH_MODEL_BASE_PATH=/DATA/disk1/yjh_space/idea2_workspace/code/FastWAM/checkpoints
```

## 8. LIBERO sim 打通（官方 vs World2WAM）

### 8.1 导出 merged checkpoint（Policy bundle → FastWAM `.pt`）

```bash
WORLD2WAM_BUNDLE=experiments/world2wam_policy_improve_full/checkpoints/world2wam_final.pt \
  bash scripts/08_export_libero_checkpoint.sh
```

输出：`experiments/exported_ckpts/world2wam_merged_*.pt`（可直接给 FastWAM LIBERO eval 使用）。

### 8.2 单次 LIBERO spatial success

```bash
# Smoke（2 tasks × 5 trials，无 tmux）
NUM_TRIALS=5 TASK_LIMIT=2 USE_TMUX=0 CUDA_DEVICES=2 \
  bash scripts/run_libero_spatial_success.sh

# World2WAM Policy（自动 export + eval）
WORLD2WAM_BUNDLE=experiments/world2wam_policy_improve_full/checkpoints/world2wam_final.pt \
  NUM_TRIALS=5 TASK_LIMIT=2 USE_TMUX=0 \
  bash scripts/run_libero_spatial_success.sh

# 全量（10 tasks × 50 trials）
NUM_TRIALS=50 CUDA_DEVICES=2,3,6,7 bash scripts/run_libero_spatial_success.sh
```

说明：使用 `scripts/libero/run_libero_manager.py` + `libero_single_task.sh`，tmux worker 会自动 `conda activate world2wam`。

### 8.3 官方 vs World2WAM 对比

```bash
# Smoke 对比
bash scripts/run_compare_libero_success.sh

# 全量对比
FULL_RUN=1 bash scripts/run_compare_libero_success.sh
```

产物：`experiments/libero_eval/compare_*/compare_summary.json`

### 8.4 Policy 全量训练 + smoke sim

```bash
bash scripts/run_full_pipeline_policy.sh
```

### 8.5 双向消融 sweep

```bash
# Smoke（验证 sweep 机制）
SMOKE=1 bash scripts/sweep_bidirectional_ablations.sh

# 全网格 mode × lambda_cycle × horizon
bash scripts/sweep_bidirectional_ablations.sh
```

产物：`experiments/ablations/summary.csv`

### 8.6 一键论文流水线

```bash
# Smoke 全流程（stage 0-2 + smoke ablation）
bash scripts/run_paper_libero_all.sh

# 含全量 compare + 全消融
FULL_RUN=1 bash scripts/run_paper_libero_all.sh

# 单 stage
STAGE=1 bash scripts/run_paper_libero_all.sh
```

后台：

```bash
bash scripts/bg_launch.sh policy_full    # Policy 全量训练 pipeline
bash scripts/bg_launch.sh libero_compare # 官方 vs World2WAM 对比
```

## 9. Robotwin（预留）

统一推理接口见 `src/eval/policy_backend.py`（`FastWAMLiberoBackend` 已可用；`FastWAMRobotwinBackend` 为 stub）。

# Cursor 提示词：新服务器部署 World2WAM 环境（Phase 1 无权重）

> 用法：整段复制到新服务器 Cursor Agent，按 Phase 1 执行。  
> 权重 ~62G 由源服务器（fiveages）晚间 rsync，见文末 Phase 2。

---

## 复制以下内容到 Cursor

```text
# 任务：在新服务器部署 World2WAM / Version A 运行环境（不含权重，权重今晚再传）

## 背景
- 项目：Physics-Aligned World2WAM
  - Version A = FlowDiT + physics router，在 frozen FastWAM pooled latent 上训练
  - Version B = MoT stack（代码在 version_b/，本次先不跑实验）
- GitHub：https://github.com/LirPan/World2WAM
- 本机 workspace：/DATA/disk0/yjh/world2wam
- 源服务器（fiveages-A100-2）：/DATA/disk0/jianhua
  - 今晚会 rsync FastWAM 权重 ~62G 到本机
- 目标：今天搭好代码、conda、LIBERO 仿真、原始数据集；**不要跑 precompute / train / eval**

## Phase 1 — 今天执行（无权重）

### 1. 检查机器
export WORKSPACE=/DATA/disk0/yjh/world2wam

df -h /DATA/disk0
nvidia-smi -L
nvidia-smi --query-gpu=index,memory.total,memory.used,utilization.gpu --format=csv
which conda || echo "need miniconda"

要求：/DATA/disk0 至少 100G 空闲（今晚权重 ~62G；以后 cache 可能 600G+）。

### 2. 克隆代码 + workspace 结构
mkdir -p "${WORKSPACE}"
cd "${WORKSPACE}"

git clone https://github.com/LirPan/World2WAM.git "Physics-Aligned World2WAM"
ln -sfn "${WORKSPACE}/Physics-Aligned World2WAM" minimal_world2wam
mkdir -p configs cache experiments cache/bg_jobs

### 3. Conda 环境
# 若无 conda：安装 miniconda 到 ~/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env create -f "${WORKSPACE}/Physics-Aligned World2WAM/scripts/world2wam_env.yaml"
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"
export MUJOCO_GL=egl

### 4. 改配置路径
bash "${WORKSPACE}/Physics-Aligned World2WAM/scripts/patch_workspace_paths.sh" "${WORKSPACE}"

确认存在：
- ${WORKSPACE}/configs/world2wam_physics_flow_dit_main.yaml
- ${WORKSPACE}/Physics-Aligned World2WAM/configs/world2wam_physics_flow_dit_main.yaml

### 5. FastWAM 目录占位（权重今晚才到）
FASTWAM="${WORKSPACE}/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM"
mkdir -p "${FASTWAM}/checkpoints"/{fastwam_release,DiffSynth-Studio,Wan-AI}
mkdir -p "${FASTWAM}/data/libero_mujoco3.3.2"
mkdir -p "${WORKSPACE}/plr/yjh_space_backup_20250602/idea2_workspace/code/LIBERO_fresh"

### 6. LIBERO 原始数据 + 仿真环境（~2G，今天可做）
# setup_deps.sh 若写死 /DATA/disk0/jianhua，先改为支持 WORKSPACE 环境变量
export WORKSPACE=/DATA/disk0/yjh/world2wam
bash "${WORKSPACE}/Physics-Aligned World2WAM/scripts/setup_deps.sh" all

若 HuggingFace 下载失败，配置 http_proxy/https_proxy 后重试。

预期：
- ${FASTWAM}/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot/ 有 parquet + mp4
- LIBERO_fresh clone 成功
- python -c "from libero.libero import benchmark; print('LIBERO OK')" 通过

### 7. 修复脚本硬编码路径（如有）
检查并修复仍写死 /DATA/disk0/jianhua 的脚本，至少包括：
- scripts/smoke_test.sh
- scripts/setup_deps.sh
- scripts/run_version_a_full_pipeline.sh（应支持 WORKSPACE 环境变量）

smoke_test 推荐写法：
  WORKSPACE="${WORKSPACE:-/DATA/disk0/yjh/world2wam}"
  source "$(conda info --base)/etc/profile.d/conda.sh"

### 8. 验证（不要求权重存在）
conda activate world2wam
export WORKSPACE=/DATA/disk0/yjh/world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"

# 配置加载
python -c "
from minimal_world2wam.utils.config import load_config
cfg = load_config('configs/world2wam_physics_flow_dit_main.yaml')
print('fastwam_root:', cfg['fastwam_root'])
print('libero_root:', cfg.get('libero_root', cfg.get('libero', {})))
"

# 单元测试（CPU）
cd "${WORKSPACE}/Physics-Aligned World2WAM"
python -m pytest tests/ -q --tb=short 2>/dev/null || true

# 数据集 5 samples
python -c "
from minimal_world2wam.utils.config import load_config
from minimal_world2wam.data.libero_transition_dataset import LiberoTransitionDataset, build_fastwam_dataset
cfg = load_config('configs/world2wam_physics_flow_dit_main.yaml')
base, _ = build_fastwam_dataset(cfg)
ds = LiberoTransitionDataset(base, horizon=cfg['horizon'], max_samples=5)
s = ds[0]
print('len', len(ds), 'obs_t', tuple(s['obs_t'].shape), 'action', tuple(s['action_chunk'].shape))
"

### 9. 写部署状态
写入 ${WORKSPACE}/cache/REMOTE_SETUP_STATUS.txt：
- 时间戳、hostname
- conda / LIBERO / 数据集 是否 OK
- 权重目录是否为空（等待 rsync）
- GPU 数量与型号
- 未完成项

## Phase 2 — 权重到达后（源服务器 fiveages 执行 rsync，本机验证）

源服务器 rsync 目标路径：
  ${WORKSPACE}/plr/.../FastWAM/checkpoints/fastwam_release/      (~12G)
  ${WORKSPACE}/plr/.../FastWAM/checkpoints/DiffSynth-Studio/    (~27G)
  ${WORKSPACE}/plr/.../FastWAM/checkpoints/Wan-AI/              (~19G)
  ${WORKSPACE}/plr/.../FastWAM/checkpoints/ActionDiT_*.pt    (~3.9G)

本机验证：
export WORKSPACE=/DATA/disk0/yjh/world2wam
ls -lh "${WORKSPACE}/plr/yjh_space_backup_20250602/idea2_workspace/code/FastWAM/checkpoints/fastwam_release/libero_uncond_2cam224.pt"
bash "${WORKSPACE}/minimal_world2wam/scripts/smoke_test.sh"

通过后启动 Version A：
export WORKSPACE=/DATA/disk0/yjh/world2wam
bash minimal_world2wam/scripts/poll_gpu_version_a.sh start
bash minimal_world2wam/scripts/poll_gpu_version_a.sh status

## 约束
- 今天不要跑 precompute / train / eval（缺权重必失败）
- 不要改模型结构或训练逻辑，只做部署
- WORKSPACE 统一为 /DATA/disk0/yjh/world2wam
- 大文件（权重、cache）靠 rsync，不从 GitHub 下权重

## 交付清单
1. 关键目录是否存在
2. conda / LIBERO / 数据集验证结果
3. GPU 信息
4. REMOTE_SETUP_STATUS.txt 全文
5. 权重到位前还缺什么
```

---

## 源服务器晚间传权重（fiveages 上执行）

直连 SSH：

```bash
bash "/DATA/disk0/jianhua/Physics-Aligned World2WAM/scripts/migrate_to_remote.sh" weights
```

若 fiveages 无法直连新服务器，笔记本开反向隧道后：

```bash
# 笔记本
FIVEAGES_HOST=<fiveages-ip> bash scripts/local_reverse_tunnel.sh

# fiveages
REMOTE_USER_HOST=yjh@127.0.0.1 REMOTE_PORT=2222 \
  bash "/DATA/disk0/jianhua/Physics-Aligned World2WAM/scripts/migrate_to_remote.sh" weights
```

---

## 可选：传完权重后再传 cache（省 precompute 时间）

300k cache ~658G：

```bash
bash "/DATA/disk0/jianhua/Physics-Aligned World2WAM/scripts/migrate_to_remote.sh" cache
```

或一键 tier1+数据+权重（不含 cache）：

```bash
bash "/DATA/disk0/jianhua/Physics-Aligned World2WAM/scripts/deploy_remote_env.sh"
```

---

## 路径速查

| 项 | 路径 |
|----|------|
| WORKSPACE | `/DATA/disk0/yjh/world2wam` |
| 代码 | `Physics-Aligned World2WAM/` |
| 主配置 | `configs/world2wam_physics_flow_dit_main.yaml` |
| FastWAM 根 | `plr/.../FastWAM/` |
| 权重 | `plr/.../FastWAM/checkpoints/` |
| LIBERO 仿真 | `plr/.../LIBERO_fresh/` |
| 原始数据 | `plr/.../FastWAM/data/libero_mujoco3.3.2/` |
| Latent cache | `cache/libero_spatial_h10_full_fastwam/` |
| Version A 启动 | `bash minimal_world2wam/scripts/poll_gpu_version_a.sh start` |

# FastWAM / LIBERO / 参考仓库代码扫描报告

> 只读扫描，未修改任何原始仓库。扫描日期：2026-05-26

---

## 1. FastWAM 主模型文件路径

| 文件 | 类 / 说明 |
|------|-----------|
| `idea2_workspace/code/FastWAM/src/fastwam/models/wan22/fastwam.py` | **`FastWAM`** — LIBERO uncond baseline（video + action 联合 flow-matching） |
| `idea2_workspace/code/FastWAM/src/fastwam/models/wan22/fastwam_joint.py` | `FastWAMJoint` — action  attend 全部 video tokens |
| `idea2_workspace/code/FastWAM/src/fastwam/models/wan22/fastwam_idm.py` | `FastWAMIDM` — teacher-forcing cond-video 分支 |
| `idea2_workspace/code/FastWAM/src/fastwam/models/wan22/wan22.py` | `Wan22Core` — 仅 video |
| `idea2_workspace/code/FastWAM/src/fastwam/runtime.py` | `create_fastwam`, `run_training`, `run_inference` |

**最小版本选用**：`FastWAM`（对应 task `libero_uncond_2cam224_1e-4`）。

---

## 2. FastWAM action head 文件路径

| 文件 | 符号 |
|------|------|
| `idea2_workspace/code/FastWAM/src/fastwam/models/wan22/action_dit.py` | **`ActionDiT`** + **`ActionHead`**（LayerNorm + Linear，`post_dit` 输出 action noise） |
| `idea2_workspace/code/FastWAM/src/fastwam/models/wan22/mot.py` | **`MoT`** — video/action 混合注意力 |

Action 预测路径：`noisy_action` → `action_expert.pre_dit` → `mot` → `action_expert.post_dit` → `ActionHead`。

---

## 3. FastWAM training loop 文件路径

| 文件 | 说明 |
|------|------|
| `idea2_workspace/code/FastWAM/scripts/train.py` | Hydra 入口 `@hydra.main` → `run_training(cfg)` |
| `idea2_workspace/code/FastWAM/src/fastwam/runtime.py` | `run_training`：instantiate model + datasets → `Wan22Trainer` |
| `idea2_workspace/code/FastWAM/src/fastwam/trainer.py` | **`Wan22Trainer.train()`**：Accelerate/DeepSpeed，`training_loss(sample)` |

训练步核心（`trainer.py` ~674）：

```python
loss, loss_dict = train_model.training_loss(sample)
```

---

## 4. FastWAM loss 计算文件路径

| 位置 | 内容 |
|------|------|
| `fastwam.py` **`training_loss`** L448–568 | `loss_video`（VAE latent flow MSE）+ `loss_action`（action flow MSE） |
| `configs/model/fastwam.yaml` | `loss.lambda_action: 1.0`（无单独 lambda_video 于 fastwam.yaml，video 默认 1.0 于模型构造） |
| 返回 | `loss_total, loss_dict` 含 `loss_video`, `loss_action` |

World2WAM 附加 loss（本项目）：`L_total = L_action + lambda_fwd * L_future`（在 `minimal_world2wam` 中实现，不改 FastWAM）。

---

## 5. FastWAM LIBERO config / script 文件路径

| 类型 | 路径 |
|------|------|
| 数据 | `configs/data/libero_2cam.yaml` — 2 cam, 33 frames, LeRobot dirs |
| Task | `configs/task/libero_uncond_2cam224_1e-4.yaml` — `model: fastwam`, lr 1e-4 |
| 训练 | `scripts/train_zero1.sh` → `scripts/train.py task=libero_uncond_2cam224_1e-4` |
| 文本缓存 | `scripts/precompute_text_embeds.py` |

数据目录（配置内相对路径）：`./data/libero_mujoco3.3.2/libero_{spatial,object,goal,10}_no_noops_lerobot`。

---

## 6. FastWAM inference / eval 文件路径

| 路径 | 说明 |
|------|------|
| `experiments/libero/eval_libero_single.py` | 单任务 LIBERO sim eval；**默认 `model.infer_action`** |
| `experiments/libero/run_libero_manager.py` | 多 GPU manager |
| `configs/sim_libero.yaml` | eval 配置；`visualize_future_video: false` 默认 |
| `FastWAM.infer_action` | `fastwam.py` L906+ — action-only（`prefill_video_cache` + action diffusion） |
| `FastWAM.infer_joint` | 联合 video+action；**推理阶段不使用** |

---

## 7. LIBERO dataset / benchmark 入口路径

| 用途 | 路径 |
|------|------|
| Benchmark | `LIBERO/libero/libero/benchmark/__init__.py` — `get_benchmark(name)(task_order_index)` |
| 原生 HDF5 训练 | `LIBERO/libero/lifelong/datasets.py` — `get_dataset`, `SequenceVLDataset` |
| Sim eval | `LIBERO/libero/lifelong/metric.py` — `raw_obs_to_tensor_obs` |

**本项目数据主路径**：FastWAM **`RobotVideoDataset`**（LeRobot 格式），非 LIBERO 原生 HDF5。

---

## 8. Batch 中可用字段（初步判断）

### FastWAM `RobotVideoDataset._get` 输出

| 字段 | Shape | 说明 |
|------|-------|------|
| `video` | `[3, T_vid, H, W]` | 归一化到 [-1,1]；LIBERO 约 T_vid=9 |
| `action` | `[T_a, 7]` | T_a=32 |
| `proprio` | `[T_a, 8]` | 与 action 对齐 |
| `prompt` | `str` | 含 task 的模板句 |
| `context` | `[L, D]` | 预计算 T5 embed |
| `context_mask` | `[L]` | |
| `image_is_pad`, `action_is_pad`, `proprio_is_pad` | mask | |

### `minimal_world2wam` adapter 扩展

| 字段 | 说明 |
|------|------|
| `language` | 复制自 `prompt` |
| `obs` | 当前帧图像（推理用，训练可从 `video[:,0]` 取） |
| `future_obs` | `video` 中未来帧 `[3,1,H,W]` |
| `future_latent` | 预计算 VAE pool 向量；无 cache 时为 `None` |
| `anchor_action_idx` | clip 内 anchor（默认 0） |

### LIBERO 原生 batch（参考）

`obs["agentview_rgb"]`, `actions`, `task_emb` — `seq_len=10`, **无 `next_obs`**。

---

## 9. 最适合作为 future latent head 输入的 hidden

**推荐**：MoT 之后 **`tokens_out["action"]`**，shape `[B, T_a, hidden_dim]`，`hidden_dim=1024`（`action_dit_config`）。

聚合方式（训练 distillation）：

- 对有效 token 做 **masked mean pool**（`~action_is_pad`）
- 或取 anchor 时刻对应 token：`idx = anchor_action_idx`

**不推荐**作为首版输入：video DiT tokens（维度过大）、原始 RGB。

Hook 挂载点：`model.mot` 的 `forward` 返回值中 `tokens_out["action"]`（wrapper 内 forward hook，不改 FastWAM 源码）。

---

## 10. future_latent target 应该从哪里来

**优先级 A（已实现路径）**：

1. 从 clip 内 `video` 取 `future_obs`（见下节索引公式）
2. **Frozen `FastWAM.vae.encode`**（`wan_video_vae.py`）
3. Global mean pool over spatial+time → 向量，**`future_latent_dim=48`**（`in_dim` / latent channels）

**索引公式**（`num_frames=33`, `action_video_freq_ratio=4`）：

- `T_vid = 9`, `T_a = 32`, `actions_per_vid = 4`
- `vid_idx_future = min((t + H) // 4 + 1, T_vid - 1)`
- `future_obs = video[:, vid_idx_future:vid_idx_future+1]`

**优先级 B**：复用 FastWAM 完整 clip 的 `_encode_video_latents` 再取未来时间片（更重，预计算脚本可选）。

**禁止**：随机 tensor、伪造 latent。

---

## 11. 在不改原仓库的情况下如何通过 wrapper 复用 FastWAM

1. `sys.path.insert` FastWAM 根目录（含 `src` 父级）
2. Hydra/OmegaConf 加载 `configs/task/libero_uncond_2cam224_1e-4.yaml` + `instantiate(cfg.model)`
3. **`FastWAMWrapper`**：
   - `forward_train` → `model.training_loss(batch)` + hook 取 hidden
   - `forward_action_only` → `model.infer_action(...)`，不调用 future head
4. 数据：`RobotVideoDataset` 经 `LiberoDatasetAdapter` 包装
5. Baseline 全量训练：subprocess 调用 `FastWAM/scripts/train_zero1.sh`（不修改其文件）
6. Future distill：仅训 `FutureLatentHead`（可选 freeze FastWAM）

---

## 参考仓库摘要

### World2VLM

- 三阶段：world model 轨迹 → A1–D4 监督 JSONL → Qwen2.5-VL SFT/GRPO
- **无** VLM←WM latent loss；可参考 shell/YAML 分阶段编排
- Latent 相关扩展点：`hy_worldplay` pipeline `prepare_latents`

### giga-world-policy

- `action_only=True`：transformer 省略 future video tokens；pipeline 跳过 VAE decode
- 对应本项目 `forward_action_only` / `eval` 中 `future_head_called=False`

---

## 维度与配置默认值

| 参数 | 值 | 来源 |
|------|-----|------|
| `action_dim` | 7 | `libero_2cam.yaml` |
| `hidden_dim` | 1024 | `fastwam.yaml` → `action_dit_config.hidden_dim` |
| `future_latent_dim` | 48 | VAE latent channel `in_dim`（mean pool） |

---

## 待手动确认的接口（TODO）

| 项 | 确认方式 |
|----|----------|
| Wan2.2 + ActionDiT 权重路径 | 首次 `FastWAMWrapper` 加载 |
| LeRobot 数据目录存在 | `lerobot_dataset_dirs` in config |
| `vae.encode` 单帧 `[B,3,1,H,W]` | `02_precompute_future_latents.sh` |
| MoT hook 在 training_loss 内触发 | 单 batch `forward_train` |
| LIBERO sim + checkpoint eval | 调 FastWAM `eval_libero_single.py` |

---

## 下一步对接清单（给实现者）

| 目标 | 文件 | 符号/字段 |
|------|------|-----------|
| 封装模型 | `fastwam.py` | `FastWAM`, `training_loss`, `infer_action` |
| Hidden | `mot.py` forward 输出 | `tokens_out["action"]` |
| Visual encoder | `wan_video_vae.py` | `model.vae.encode` |
| 训练 batch | `robot_video_dataset.py` | `video`, `action`, `context`, `*_is_pad` |
| LIBERO eval | `eval_libero_single.py` | `infer_action` |

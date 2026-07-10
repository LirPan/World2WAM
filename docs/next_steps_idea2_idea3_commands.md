# World2WAM 下一步命令

工作目录：`/DATA/disk0/jianhua`（通过软链接 `minimal_world2wam` 访问代码）

## 1. 检查 cache shape

```bash
python minimal_world2wam/scripts/inspect_cache_shapes.py \
  --cache_dir cache/libero_spatial_h10_full_fastwam \
  --num_samples 10
```

## 2. Cache-only offline eval（不加载 FastWAM）

```bash
python minimal_world2wam/eval/eval_offline_cache_only.py \
  --config configs/world2wam_libero_spatial_h10_paper.yaml \
  --cache_dir cache/libero_spatial_h10_full_fastwam \
  --heads_ckpt experiments/world2wam_heads_paper/heads_final.pt \
  --adapter_ckpt experiments/world2wam_adapter_paper/adapter_final.pt \
  --output experiments/eval_offline_paper_300k.json \
  --max_samples 50000
```

## 3. Stage2 warm-start 训练 adapter

```bash
python minimal_world2wam/train/train_world2wam_adapter.py \
  --config configs/world2wam_libero_spatial_h10_paper.yaml \
  --cache_dir cache/libero_spatial_h10_full_fastwam \
  --warm_start_heads experiments/world2wam_heads_paper/heads_final.pt \
  --output_dir experiments/world2wam_adapter_warmstart_paper \
  --use_act true --use_fwd true --use_inv true --use_cycle true
```

## 4. 检查 physics 伪标签分布

```bash
python minimal_world2wam/scripts/inspect_physics_labels.py \
  --cache_dir cache/libero_spatial_h10_full_fastwam \
  --num_samples 5000 \
  --output experiments/physics_label_histogram.json
```

## 5. idea3 debug 训练（300 steps）

```bash
python minimal_world2wam/train/train_physics_world2wam.py \
  --config configs/world2wam_idea3_physics_spatial_h10.yaml \
  --cache_dir cache/libero_spatial_h10_full_fastwam \
  --heads_ckpt experiments/world2wam_heads_paper/heads_final.pt \
  --adapter_ckpt experiments/world2wam_adapter_paper/adapter_final.pt \
  --output_dir experiments/world2wam_idea3_physics_debug \
  --max_samples 5000 \
  --max_steps 300 \
  --use_act true \
  --use_physics_router true \
  --use_physics_losses true
```

## 6. Residual adapter sim eval（需 GPU + LIBERO）

```bash
python minimal_world2wam/eval/eval_libero_world2wam.py \
  --config configs/world2wam_libero_spatial_h10_paper.yaml \
  --mode ours_residual \
  --adapter_ckpt experiments/world2wam_adapter_paper/adapter_final.pt \
  --residual_alpha 0.1 \
  --max_tasks 1 --num_trials 1 \
  --output experiments/eval_ours_residual.json
```

## 7. MLP adapter sim eval

```bash
python minimal_world2wam/eval/eval_libero_world2wam.py \
  --config configs/world2wam_libero_spatial_h10_paper.yaml \
  --mode ours_adapter \
  --adapter_ckpt experiments/world2wam_adapter_paper/adapter_final.pt \
  --max_tasks 1 --num_trials 1 \
  --output experiments/eval_ours_adapter.json
```

## 8. Light DiT adapter debug 训练（300 steps）

```bash
python minimal_world2wam/train/train_world2wam_adapter.py \
  --config configs/world2wam_libero_spatial_h10_dit.yaml \
  --cache_dir cache/libero_spatial_h10_full_fastwam \
  --warm_start_heads experiments/world2wam_heads_paper/heads_final.pt \
  --output_dir experiments/world2wam_adapter_dit_debug \
  --adapter_type light_dit \
  --max_samples 5000 \
  --max_steps 300 \
  --use_act true --use_fwd true --use_inv true --use_cycle true \
  --device cpu
```

## 9. idea3 + Light DiT debug

```bash
python minimal_world2wam/train/train_physics_world2wam.py \
  --config configs/world2wam_idea3_physics_spatial_h10.yaml \
  --cache_dir cache/libero_spatial_h10_full_fastwam \
  --heads_ckpt experiments/world2wam_heads_paper/heads_final.pt \
  --adapter_ckpt experiments/world2wam_adapter_paper/adapter_final.pt \
  --output_dir experiments/world2wam_idea3_physics_dit_debug \
  --adapter_type light_dit \
  --max_samples 5000 --max_steps 300 \
  --use_act true --use_physics_router true --use_physics_losses true \
  --device cpu
```

## 10. Offline eval Light DiT

```bash
python minimal_world2wam/eval/eval_offline_cache_only.py \
  --config configs/world2wam_libero_spatial_h10_dit.yaml \
  --cache_dir cache/libero_spatial_h10_full_fastwam \
  --heads_ckpt experiments/world2wam_heads_paper/heads_final.pt \
  --adapter_ckpt experiments/world2wam_adapter_dit_debug/adapter_final.pt \
  --adapter_type light_dit \
  --output experiments/eval_offline_dit_debug.json \
  --max_samples 5000 --device cpu
```

## 11. Sim eval Light DiT one-step（需 GPU）

```bash
python minimal_world2wam/eval/eval_libero_world2wam.py \
  --config configs/world2wam_libero_spatial_h10_dit.yaml \
  --mode ours_onestep_dit \
  --adapter_ckpt experiments/world2wam_adapter_dit_debug/adapter_final.pt \
  --adapter_type light_dit \
  --max_tasks 1 --num_trials 1 \
  --output experiments/eval_ours_onestep_dit.json
```

## 12. Sim eval residual Light DiT（需 GPU）

```bash
python minimal_world2wam/eval/eval_libero_world2wam.py \
  --config configs/world2wam_libero_spatial_h10_dit.yaml \
  --mode ours_residual_dit \
  --adapter_ckpt experiments/world2wam_adapter_dit_debug/adapter_final.pt \
  --adapter_type light_dit \
  --residual_alpha 0.1 \
  --max_tasks 1 --num_trials 1 \
  --output experiments/eval_ours_residual_dit.json
```

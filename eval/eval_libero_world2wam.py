#!/usr/bin/env python3
"""LIBERO evaluation for World2WAM.

- baseline: official FastWAM action-only (delegates to FastWAM eval_libero_single.py)
- ours_adapter: MLP ActionAdapter — z_t + text -> action (one forward pass)
- ours_dit: FastWAM Action DiT — frozen infer_action() flow-matching diffusion (idea2 design path)
- offline: latency smoke on encoder + optional latent_verification on cached latents

ForwardHead / InverseHead remain training-only; they are never used in sim control.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))


def _resolve_path(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = (WORKSPACE / path).resolve()
    return path

from minimal_world2wam.data.latent_cache_dataset import LatentCacheDataset, collate_latent_batch, load_meta
from minimal_world2wam.eval.physics_eval_utils import build_physics_model, load_physics_checkpoint
from minimal_world2wam.models.world2wam_heads import build_heads_from_config, resolve_adapter_type
from minimal_world2wam.physics.physics_labels import PHYSICS_PHASES
from minimal_world2wam.train.training_utils import (
    call_action_adapter,
    is_flow_adapter,
    load_checkpoint,
    sample_action_adapter,
)
from minimal_world2wam.utils.config import load_config
from minimal_world2wam.utils.seed import set_seed
from minimal_world2wam.wrappers.fastwam_encoder import add_fastwam_path, load_frozen_fastwam
from minimal_world2wam.wrappers.inference_guard import inference_guard


def _libero_env(cfg: dict) -> dict:
    env = os.environ.copy()
    extra_paths: list[str] = []
    libero_root = _resolve_path(cfg.get("libero", {}).get("root"))
    if libero_root and libero_root.is_dir():
        extra_paths.append(str(libero_root))
    src_root = WORKSPACE / "cache" / "src"
    for name in ("robosuite-1.4.0", "bddl-1.0.1", "bddl-100-1.0.1"):
        p = src_root / name
        if p.is_dir():
            extra_paths.append(str(p))
    if extra_paths:
        prefix = ":".join(extra_paths)
        env["PYTHONPATH"] = f"{prefix}:{env.get('PYTHONPATH', '')}".rstrip(":")
    env.setdefault("MUJOCO_GL", "egl")
    return env


def _ensure_libero_importable(cfg: dict) -> None:
    for p in _libero_env(cfg).get("PYTHONPATH", "").split(":"):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def run_baseline_sim(
    cfg: dict,
    *,
    max_tasks: int = 1,
    num_trials: int = 1,
    device_override: str | None = None,
) -> dict:
    """Delegate to official FastWAM LIBERO eval (one subprocess per task)."""
    import re

    fastwam_root = Path(cfg["fastwam_root"])
    eval_script = fastwam_root / "experiments/libero/eval_libero_single.py"
    if not eval_script.is_file():
        raise FileNotFoundError(f"Missing {eval_script}")

    ckpt = _resolve_path(cfg["official_fastwam_checkpoint"])
    stats = _resolve_path(cfg["dataset_stats_path"])
    per_task: list[dict] = []
    total_success = 0
    total_episodes = 0

    for task_id in range(int(max_tasks)):
        cmd = [
            sys.executable,
            str(eval_script),
            f"ckpt={ckpt}",
            f"EVALUATION.dataset_stats_path={stats}",
            f"EVALUATION.task_id={task_id}",
            f"EVALUATION.num_trials={num_trials}",
        ]
        if device_override:
            cmd.append(f"EVALUATION.device={device_override}")
        print(f"Running baseline task {task_id}/{max_tasks - 1}:", " ".join(cmd))
        proc = subprocess.run(
            ["xvfb-run", "-a", *cmd],
            cwd=str(fastwam_root),
            capture_output=True,
            text=True,
            env=_libero_env(cfg),
        )
        print(proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout)
        if proc.returncode != 0:
            print(proc.stderr[-2000:] if proc.stderr else "")
            raise RuntimeError(f"Baseline eval failed for task {task_id} with code {proc.returncode}")

        task_successes = task_total = 0
        m = re.search(r"Task \d+ completed: (\d+)/(\d+) successes", proc.stdout)
        if m:
            task_successes, task_total = int(m.group(1)), int(m.group(2))
        per_task.append(
            {
                "task_id": task_id,
                "successes": task_successes,
                "trials": task_total or num_trials,
                "returncode": proc.returncode,
            }
        )
        total_success += task_successes
        total_episodes += task_total or num_trials

    return {
        "mode": "baseline",
        "suite": str(cfg.get("suite", "libero_spatial")),
        "max_tasks": int(max_tasks),
        "num_trials": int(num_trials),
        "successes": total_success,
        "total_episodes": total_episodes,
        "success_rate": float(total_success) / max(total_episodes, 1),
        "per_task": per_task,
    }


def measure_infer_latency(
    encoder,
    device: str,
    cache_dir: Path | None = None,
    n_warmup: int = 3,
    n_iter: int = 10,
) -> dict:
    dummy = torch.randn(1, 3, 224, 448)
    batch: dict = {"obs_t": dummy, "action_horizon": 10}
    if cache_dir is not None:
        cache_path = cache_dir / "000000.pt"
        if cache_path.is_file():
            sample = torch.load(cache_path, map_location="cpu", weights_only=False)
            text_embed = sample["text_embed"]
            if text_embed.dim() == 2:
                text_embed = text_embed.unsqueeze(0)
            batch["context"] = text_embed
            batch["context_mask"] = torch.ones(text_embed.shape[0], text_embed.shape[1], dtype=torch.bool)
        else:
            batch["prompt"] = "pick up the bowl"
    else:
        batch["prompt"] = "pick up the bowl"
    with inference_guard():
        for _ in range(n_warmup):
            encoder.infer_action_only(batch)
        t0 = time.perf_counter()
        for _ in range(n_iter):
            encoder.infer_action_only(batch)
        elapsed = (time.perf_counter() - t0) / n_iter
    return {"infer_latency_ms": elapsed * 1000, "n_iter": n_iter}


def latent_verification(cfg: dict, cache_dir: Path, heads_ckpt: Path | None, device: str) -> dict:
    meta = load_meta(cache_dir)
    ds = LatentCacheDataset(cache_dir)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_latent_batch)
    heads = build_heads_from_config(cfg, meta, include_inverse=False)
    forward_head = heads["forward"].to(device)
    action_adapter = heads["adapter"].to(device)
    if heads_ckpt and heads_ckpt.is_file():
        load_checkpoint(heads_ckpt, forward_head, action_adapter)
    forward_head.eval()
    action_adapter.eval()
    flow_steps = int(cfg.get("eval", {}).get("flow_sample_steps", 10))

    fwd_sum = inv_sum = cycle_sum = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            z_pred = forward_head(batch["z_t"], batch["action_chunk"], batch["text_embed"])
            if is_flow_adapter(action_adapter):
                pred_a = sample_action_adapter(
                    action_adapter,
                    batch["z_t"],
                    batch["text_embed"],
                    num_steps=flow_steps,
                    future_latent=batch["z_tH"],
                )
                a_cycle = sample_action_adapter(
                    action_adapter,
                    batch["z_t"],
                    batch["text_embed"],
                    num_steps=flow_steps,
                    future_latent=z_pred,
                )
            else:
                pred_a = call_action_adapter(action_adapter, batch["z_t"], batch["text_embed"])
                a_cycle = pred_a
            fwd_sum += torch.nn.functional.mse_loss(z_pred, batch["z_tH"]).item()
            inv_sum += torch.nn.functional.mse_loss(pred_a, batch["action_chunk"]).item()
            cycle_sum += torch.nn.functional.mse_loss(a_cycle, batch["action_chunk"]).item()
            n += 1
    return {
        "latent_verification": True,
        "mse_fwd": fwd_sum / max(n, 1),
        "mse_inv": inv_sum / max(n, 1),
        "mse_cycle": cycle_sum / max(n, 1),
        "batches": n,
    }


def _load_libero_eval_deps(cfg: dict):
    _ensure_libero_importable(cfg)
    fastwam_root = Path(cfg["fastwam_root"]).resolve()
    for p in (fastwam_root, fastwam_root / "src"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
    from experiments.libero.libero_utils import (  # type: ignore
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        invert_gripper_action,
        quat2axisangle,
    )
    from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor  # type: ignore
    from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # type: ignore
    from hydra import compose, initialize_config_dir  # type: ignore
    from hydra.core.global_hydra import GlobalHydra  # type: ignore
    from hydra.utils import instantiate  # type: ignore
    from libero.libero import benchmark  # type: ignore

    return (
        benchmark,
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        invert_gripper_action,
        quat2axisangle,
        FastWAMProcessor,
        load_dataset_stats_from_json,
        GlobalHydra,
        initialize_config_dir,
        compose,
        instantiate,
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _extract_sim_state(obs: dict, quat2axisangle) -> np.ndarray:
    return np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)


# Match FastWAM official LIBERO eval prompt template.
_DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)


def _format_official_prompt(task_desc: str) -> str:
    return _DEFAULT_PROMPT.format(task=task_desc)


def _normalize_proprio(proprio: np.ndarray, processor) -> np.ndarray:
    """Match FastWAM eval_libero_single._normalize_proprio (returns float32 [D])."""
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    out = state_batch["state"][state_key]
    if torch.is_tensor(out):
        out = out.detach().cpu().numpy()
    return np.asarray(out, dtype=np.float32).reshape(-1)


def _build_fastwam_processor(cfg: dict):
    (
        _benchmark,
        _dummy,
        _get_env,
        _get_image,
        _invert,
        _quat2axisangle,
        _FastWAMProcessor,
        load_dataset_stats_from_json,
        GlobalHydra,
        initialize_config_dir,
        compose,
        instantiate,
    ) = _load_libero_eval_deps(cfg)
    fastwam_root = Path(cfg["fastwam_root"])
    config_dir = fastwam_root / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        hydra_cfg = compose(config_name="train", overrides=[f"task={cfg['fastwam_task_config']}"])
    dataset_stats = load_dataset_stats_from_json(str(cfg["dataset_stats_path"]))
    processor = instantiate(hydra_cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    return processor, hydra_cfg


def _denormalize_action(action: torch.Tensor, processor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(f"Expected a single action key, got {len(action_meta)}")
    action_key = action_meta[0]["key"]
    normalizer = processor.normalizer.normalizers["action"][action_key]
    denorm = normalizer.backward(action.detach().float().cpu())
    return denorm.numpy()


def _apply_fastwam_device(cfg: dict, device: str | None) -> dict:
    cfg = dict(cfg)
    if device:
        cfg["fastwam"] = dict(cfg.get("fastwam", {}))
        cfg["fastwam"]["device"] = device
    return cfg


def _num_inference_steps(cfg: dict, args) -> int:
    if getattr(args, "num_inference_steps", None) is not None:
        return int(args.num_inference_steps)
    # Official FastWAM train.yaml: eval_num_inference_steps=10
    return int(cfg.get("eval", {}).get("num_inference_steps", 10))


def _postprocess_sim_action(
    action: torch.Tensor,
    processor,
    invert_gripper_action,
    *,
    binarize_gripper: bool = True,
) -> np.ndarray:
    """Match FastWAM official sim postprocess (denorm → *2-1 → invert → optional binarize)."""
    action = _denormalize_action(action, processor)[0]
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if binarize_gripper:
        action[..., -1] = np.sign(action[..., -1])
    return action


def _parse_task_ids(args, n_tasks: int) -> list[int]:
    """Resolve task ids: --task_ids wins over 0..max_tasks-1."""
    raw = getattr(args, "task_ids", None)
    if raw:
        ids = [int(x.strip()) for x in str(raw).split(",") if x.strip() != ""]
        bad = [i for i in ids if i < 0 or i >= n_tasks]
        if bad:
            raise ValueError(f"--task_ids out of range for suite ({n_tasks} tasks): {bad}")
        return ids
    max_tasks = min(int(args.max_tasks), n_tasks)
    return list(range(max_tasks))


def _parse_adapter_phases(args) -> set[str] | None:
    raw = getattr(args, "adapter_phases", None)
    if not raw:
        return None
    phases = {p.strip() for p in str(raw).split(",") if p.strip()}
    unknown = phases - set(PHYSICS_PHASES)
    if unknown:
        raise ValueError(f"--adapter_phases unknown: {sorted(unknown)}; valid={PHYSICS_PHASES}")
    return phases


def _effective_residual_alpha(
    base_alpha: float,
    *,
    residual_gate: str = "none",
    confidence: torch.Tensor | float | None = None,
    a_fw: torch.Tensor | None = None,
    a_adapter: torch.Tensor | None = None,
    gate_confidence_thr: float = 0.4,
    gate_disagree_thr: float = 0.5,
    adapter_phases: set[str] | None = None,
    phase_name: str | None = None,
    zero_alpha_on_uncertain: bool = False,
) -> float:
    """Lean FastWAM when unsure / disagreeing / wrong phase.

    Blend convention (absolute-action adapter):
      a = (1 - α_eff) * a_fw + α_eff * a_adapter
    """
    alpha = float(base_alpha)
    if alpha <= 0.0:
        return 0.0

    if zero_alpha_on_uncertain and phase_name == "uncertain":
        return 0.0
    if adapter_phases is not None and (phase_name is None or phase_name not in adapter_phases):
        return 0.0

    gate = (residual_gate or "none").lower()
    if gate in ("none", "off", ""):
        return alpha

    if gate in ("confidence_soft", "conf_soft"):
        if confidence is None:
            return alpha
        conf = float(confidence.detach().mean().item()) if torch.is_tensor(confidence) else float(confidence)
        return max(0.0, min(alpha, alpha * conf))

    if gate in ("confidence_hard", "conf_hard"):
        if confidence is None:
            return alpha
        conf = float(confidence.detach().mean().item()) if torch.is_tensor(confidence) else float(confidence)
        return alpha if conf >= float(gate_confidence_thr) else 0.0

    if gate in ("disagreement", "disagree"):
        if a_fw is None or a_adapter is None:
            return alpha
        # Mean abs action disagreement; shrink alpha when adapter drifts from FastWAM.
        disagree = (a_adapter.to(a_fw.device) - a_fw).abs().mean().item()
        thr = max(float(gate_disagree_thr), 1e-6)
        scale = max(0.0, 1.0 - float(disagree) / thr)
        return alpha * scale

    raise ValueError(
        f"Unknown --residual_gate={residual_gate!r}; "
        "use none|confidence_soft|confidence_hard|disagreement"
    )


def _run_libero_rollout(
    cfg: dict,
    args,
    *,
    mode: str,
    predict_action_chunk,
    result_extra: dict | None = None,
) -> dict:
    """Shared LIBERO sim loop; predict_action_chunk returns (action[H,7] numpy, latency_ms)."""
    _ensure_libero_importable(cfg)
    (
        benchmark,
        get_libero_dummy_action,
        get_libero_env,
        get_libero_image,
        invert_gripper_action,
        quat2axisangle,
        _FastWAMProcessor,
        _load_dataset_stats_from_json,
        _GlobalHydra,
        _initialize_config_dir,
        _compose,
        _instantiate,
    ) = _load_libero_eval_deps(cfg)
    processor, hydra_cfg = _build_fastwam_processor(cfg)

    suite_name = str(cfg.get("suite", "libero_spatial"))
    benchmark_dict = benchmark.get_benchmark_dict()
    if suite_name not in benchmark_dict:
        raise ValueError(f"Unknown LIBERO suite {suite_name}")
    suite = benchmark_dict[suite_name]()
    task_ids = _parse_task_ids(args, suite.n_tasks)
    max_tasks = len(task_ids)
    num_trials = int(args.num_trials)
    max_steps = int(args.max_steps)
    warmup_steps = int(args.warmup_steps)
    horizon = int(cfg.get("horizon", 10))
    # Official FastWAM sim_libero.yaml: replan_steps=10 (execute first K of chunk, then replan).
    replan_steps = int(getattr(args, "replan_steps", None) or (cfg.get("eval") or {}).get("replan_steps", 10))
    replan_steps = max(1, min(replan_steps, horizon))

    video_size = hydra_cfg.data.train.get("video_size", [224, 448])
    input_h, input_w = int(video_size[0]), int(video_size[1])
    concat_mode = str(hydra_cfg.data.train.get("concat_multi_camera", "horizontal"))

    total_success = 0
    total_episodes = 0
    total_steps = 0
    infer_times_ms: list[float] = []
    per_task: list[dict] = []

    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = list(suite.get_task_init_states(task_id))
        while len(init_states) < num_trials:
            init_states.extend(init_states[: (num_trials - len(init_states))])
        env, task_desc = get_libero_env(task, 256, int(cfg.get("train", {}).get("seed", 42)))

        task_success = 0
        for trial in range(num_trials):
            env.reset()
            obs = env.set_init_state(init_states[trial])
            pending_actions: list[list[float]] = []
            step_count = 0
            done = False

            while step_count < (max_steps + warmup_steps):
                if step_count < warmup_steps:
                    obs, _, done, _ = env.step(get_libero_dummy_action())
                    step_count += 1
                    if done:
                        break
                    continue

                if len(pending_actions) == 0:
                    rgb = _prepare_obs_frame(obs, get_libero_image, input_w, input_h, concat_mode)
                    # Official FastWAM: DEFAULT_PROMPT + normalized proprio (critical for ~96% SR).
                    proprio_raw = _extract_sim_state(obs, quat2axisangle)
                    proprio = _normalize_proprio(proprio_raw, processor)
                    prompt = _format_official_prompt(task_desc)
                    action, latency_ms = predict_action_chunk(
                        rgb=rgb,
                        obs=obs,
                        task_desc=prompt,
                        proprio=proprio,
                        horizon=horizon,
                    )
                    infer_times_ms.append(latency_ms)
                    pending_actions = action[:replan_steps].tolist()

                obs, _, done, _ = env.step(pending_actions.pop(0))
                step_count += 1
                if done:
                    break

            total_episodes += 1
            total_steps += step_count
            if done:
                total_success += 1
                task_success += 1

        per_task.append({"task_id": task_id, "task_desc": task_desc, "successes": task_success, "trials": num_trials})
        env.close()

    out = {
        "mode": mode,
        "suite": suite_name,
        "max_tasks": max_tasks,
        "task_ids": task_ids,
        "num_trials": num_trials,
        "warmup_steps": warmup_steps,
        "replan_steps": replan_steps,
        "success_rate": float(total_success) / max(total_episodes, 1),
        "successes": total_success,
        "total_episodes": total_episodes,
        "average_episode_length": float(total_steps) / max(total_episodes, 1),
        "inference_latency_ms": float(np.mean(infer_times_ms)) if infer_times_ms else None,
        "per_task": per_task,
    }
    if result_extra:
        out.update(result_extra)
    return out


def _prepare_obs_frame(obs: dict, get_libero_image, width: int, height: int, concat_multi_camera: str) -> np.ndarray:
    imgs = get_libero_image(obs)
    primary_w = width // 2 if concat_multi_camera == "horizontal" else width
    primary_h = height if concat_multi_camera == "horizontal" else height // 2
    primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    wrist = _center_crop_resize(imgs["wrist_image"], width=primary_w, height=primary_h)
    if concat_multi_camera == "horizontal":
        rgb = np.concatenate([primary, wrist], axis=1)
    elif concat_multi_camera == "vertical":
        rgb = np.concatenate([primary, wrist], axis=0)
    else:
        raise ValueError(f"Unsupported concat_multi_camera={concat_multi_camera}")
    return rgb


def _load_adapter_for_eval(cfg: dict, ckpt_path: Path, adapter_type_override: str | None = None):
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    adapter_type = resolve_adapter_type(cfg, payload, adapter_type_override)
    cfg = dict(cfg)
    cfg.setdefault("model", {}).setdefault("action_adapter", {})["adapter_type"] = adapter_type
    heads = build_heads_from_config(cfg, None, include_inverse=False)
    adapter = heads["adapter"]
    load_checkpoint(
        ckpt_path,
        heads["forward"],
        adapter,
        expected_adapter_type=adapter_type,
    )
    return adapter, adapter_type


def run_ours_onestep_sim(cfg: dict, args, *, adapter_type: str, mode: str) -> dict:
    """One-step policy: a = Adapter(z_t, text) with MLP, LightActionDiT, or FlowActionDiT."""
    cfg = _apply_fastwam_device(cfg, args.device)
    deps = _load_libero_eval_deps(cfg)
    invert_gripper_action = deps[4]  # deps: ..., image, invert_gripper, quat2axisangle, ...
    processor, _hydra_cfg = _build_fastwam_processor(cfg)

    cfg = dict(cfg)
    cfg["fastwam"] = dict(cfg.get("fastwam", {}))
    cfg["fastwam"]["load_text_encoder"] = True
    encoder = load_frozen_fastwam(cfg)

    device = cfg.get("fastwam", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if not args.adapter_ckpt:
        raise ValueError(f"--adapter_ckpt is required for --mode {mode}")
    ckpt_path = _resolve_path(args.adapter_ckpt)
    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(f"adapter_ckpt not found: {args.adapter_ckpt}")

    override = getattr(args, "adapter_type", None) or adapter_type
    adapter, resolved_type = _load_adapter_for_eval(cfg, ckpt_path, override)
    adapter = adapter.to(device).eval()
    flow_sample_steps = int(getattr(args, "flow_sample_steps", 10) or 10)
    use_flow = is_flow_adapter(adapter)

    def predict_action_chunk(*, rgb, obs, task_desc, proprio, horizon):
        del obs, proprio
        obs_tensor = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        obs_tensor = obs_tensor * (2.0 / 255.0) - 1.0
        z_t = encoder.encode_obs_latent(obs_tensor)
        text_embed, _text_mask = encoder.encode_text_from_prompt(task_desc)
        text_embed = text_embed.to(device=device)

        t0 = time.perf_counter()
        with inference_guard(), torch.no_grad():
            if use_flow:
                pred_action = sample_action_adapter(
                    adapter,
                    z_t.to(device),
                    text_embed,
                    num_steps=flow_sample_steps,
                )
            else:
                pred_action = call_action_adapter(adapter, z_t.to(device), text_embed)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        action = _postprocess_sim_action(
            pred_action,
            processor,
            invert_gripper_action,
            binarize_gripper=bool(getattr(args, "binarize_gripper", True)),
        )
        return action, latency_ms

    result_extra = {
        "strategy": "onestep_adapter",
        "adapter_type": resolved_type,
        "adapter_ckpt": str(ckpt_path),
    }
    if use_flow:
        result_extra["flow_sample_steps"] = flow_sample_steps

    return _run_libero_rollout(
        cfg,
        args,
        mode=mode,
        predict_action_chunk=predict_action_chunk,
        result_extra=result_extra,
    )


def run_ours_residual_variant_sim(cfg: dict, args, *, adapter_type: str, mode: str) -> dict:
    """Residual over FastWAM.

    blend (Version A): a = (1-α)*a_FW + α*a_adapter   (adapter predicts absolute actions)
    additive (Version C): a = a_FW + α*δ               (adapter predicts δ = a* - a_FW)
    """
    cfg = _apply_fastwam_device(cfg, args.device)
    deps = _load_libero_eval_deps(cfg)
    invert_gripper_action = deps[4]  # deps: ..., image, invert_gripper, quat2axisangle, ...
    processor, _hydra_cfg = _build_fastwam_processor(cfg)

    cfg = dict(cfg)
    cfg["fastwam"] = dict(cfg.get("fastwam", {}))
    cfg["fastwam"]["load_text_encoder"] = True
    encoder = load_frozen_fastwam(cfg)

    device = cfg.get("fastwam", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    num_steps = _num_inference_steps(cfg, args)
    residual_alpha = float(getattr(args, "residual_alpha", 0.1))
    residual_gate = str(getattr(args, "residual_gate", "none") or "none")
    gate_disagree_thr = float(getattr(args, "gate_disagree_thr", 0.5))
    residual_mode = str(
        getattr(args, "residual_mode", None)
        or (cfg.get("eval") or {}).get("residual_mode")
        or "blend"
    ).lower()
    if residual_mode not in ("blend", "additive"):
        raise ValueError(f"--residual_mode must be blend|additive, got {residual_mode!r}")

    if not args.adapter_ckpt:
        raise ValueError(f"--adapter_ckpt is required for --mode {mode}")
    ckpt_path = _resolve_path(args.adapter_ckpt)
    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(f"adapter_ckpt not found: {args.adapter_ckpt}")

    override = getattr(args, "adapter_type", None) or adapter_type
    adapter, resolved_type = _load_adapter_for_eval(cfg, ckpt_path, override)
    adapter = adapter.to(device).eval()
    flow_sample_steps = int(getattr(args, "flow_sample_steps", 10) or 10)
    use_flow = is_flow_adapter(adapter)
    alpha_eff_hist: list[float] = []

    def predict_action_chunk(*, rgb, obs, task_desc, proprio, horizon):
        del obs
        obs_tensor = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        obs_tensor = obs_tensor * (2.0 / 255.0) - 1.0
        proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).float()

        batch_dit = {
            "obs_t": obs_tensor,
            "prompt": task_desc,
            "action_horizon": int(horizon),
            "proprio": proprio_tensor,
            "num_inference_steps": num_steps,
        }

        t0 = time.perf_counter()
        with inference_guard(), torch.no_grad():
            a_fastwam = encoder.infer_action_only(batch_dit)
            z_t = encoder.encode_obs_latent(obs_tensor)
            text_embed, _text_mask = encoder.encode_text_from_prompt(task_desc)
            text_embed = text_embed.to(device=device)
            if use_flow:
                adapter_out = sample_action_adapter(
                    adapter,
                    z_t.to(device),
                    text_embed,
                    num_steps=flow_sample_steps,
                )
            else:
                adapter_out = call_action_adapter(adapter, z_t.to(device), text_embed)
            a_fw = a_fastwam.to(device)
            a_ad = adapter_out.to(device)
            alpha_eff = _effective_residual_alpha(
                residual_alpha,
                residual_gate=residual_gate,
                a_fw=a_fw,
                a_adapter=a_ad if residual_mode == "blend" else (a_fw + a_ad),
                gate_disagree_thr=gate_disagree_thr,
            )
            alpha_eff_hist.append(float(alpha_eff))
            if residual_mode == "additive":
                pred_action = a_fw + alpha_eff * a_ad
            else:
                pred_action = (1.0 - alpha_eff) * a_fw + alpha_eff * a_ad
        latency_ms = (time.perf_counter() - t0) * 1000.0
        action = _postprocess_sim_action(
            pred_action,
            processor,
            invert_gripper_action,
            binarize_gripper=bool(getattr(args, "binarize_gripper", True)),
        )
        return action, latency_ms

    result_extra = {
        "strategy": "residual_adapter",
        "residual_mode": residual_mode,
        "adapter_type": resolved_type,
        "residual_alpha": residual_alpha,
        "residual_gate": residual_gate,
        "gate_disagree_thr": gate_disagree_thr,
        "adapter_ckpt": str(ckpt_path),
        "num_inference_steps": num_steps,
    }
    if use_flow:
        result_extra["flow_sample_steps"] = flow_sample_steps

    results = _run_libero_rollout(
        cfg,
        args,
        mode=mode,
        predict_action_chunk=predict_action_chunk,
        result_extra=result_extra,
    )
    if alpha_eff_hist:
        results["mean_alpha_eff"] = float(np.mean(alpha_eff_hist))
        results["alpha_eff_n"] = len(alpha_eff_hist)
    return results


def _load_physics_model_for_eval(cfg: dict, ckpt_path: Path, adapter_type_override: str | None = None):
    cfg = dict(cfg)
    if adapter_type_override:
        cfg.setdefault("model", {}).setdefault("action_adapter", {})["adapter_type"] = adapter_type_override
    device = cfg.get("fastwam", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    # Must match training: StudentPhysicsRouter in_dim = latent + text_dim + state_dim.
    # Without cache_dir, state_dim defaults to 0 (560) but ckpt was trained with state_dim=8 (568).
    cache_dir = _resolve_path((cfg.get("cache") or {}).get("output_dir"))
    meta = None
    if cache_dir is not None and cache_dir.is_dir():
        try:
            meta = load_meta(cache_dir)
        except FileNotFoundError:
            meta = None
    model, adapter_type = build_physics_model(cfg, meta, device, cache_dir=cache_dir)
    load_physics_checkpoint(model, ckpt_path, expected_adapter_type=adapter_type_override or adapter_type)
    return model, adapter_type


def run_ours_physics_onestep_sim(cfg: dict, args, *, adapter_type: str, mode: str) -> dict:
    """Physics-conditioned one-step: router inference + adapter."""
    cfg = _apply_fastwam_device(cfg, args.device)
    deps = _load_libero_eval_deps(cfg)
    invert_gripper_action = deps[4]  # deps: ..., image, invert_gripper, quat2axisangle, ...
    processor, _hydra_cfg = _build_fastwam_processor(cfg)

    cfg = dict(cfg)
    cfg["fastwam"] = dict(cfg.get("fastwam", {}))
    cfg["fastwam"]["load_text_encoder"] = True
    encoder = load_frozen_fastwam(cfg)

    device = cfg.get("fastwam", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if not args.adapter_ckpt:
        raise ValueError(f"--adapter_ckpt is required for --mode {mode}")
    ckpt_path = _resolve_path(args.adapter_ckpt)
    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(f"adapter_ckpt not found: {args.adapter_ckpt}")

    override = getattr(args, "adapter_type", None) or adapter_type
    physics_model, resolved_type = _load_physics_model_for_eval(cfg, ckpt_path, override)
    flow_sample_steps = int(getattr(args, "flow_sample_steps", 10) or 10)
    phase_label_version = getattr(args, "phase_label_version", None) or cfg.get("physics", {}).get(
        "phase_label_version", "v1"
    )
    last_phase_prob = None
    last_pred_phase = None

    def predict_action_chunk(*, rgb, obs, task_desc, proprio, horizon):
        del obs, horizon
        obs_tensor = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        obs_tensor = obs_tensor * (2.0 / 255.0) - 1.0
        z_t = encoder.encode_obs_latent(obs_tensor)
        text_embed, _text_mask = encoder.encode_text_from_prompt(task_desc)
        text_embed = text_embed.to(device=device)
        state_t = torch.from_numpy(proprio).unsqueeze(0).float().to(device) if proprio is not None else None

        t0 = time.perf_counter()
        with inference_guard(), torch.no_grad():
            out = physics_model.forward_inference(
                z_t.to(device),
                text_embed,
                state_t=state_t,
                num_flow_steps=flow_sample_steps,
            )
            pred_action = out.get("pred_action")
            if pred_action is None:
                raise RuntimeError("Physics model did not return pred_action")
            nonlocal last_phase_prob, last_pred_phase
            probs = out.get("phase_prob")
            if probs is None:
                probs = out.get("physics_probs")
            if probs is not None:
                last_phase_prob = probs[0].detach().cpu().tolist()
                last_pred_phase = PHYSICS_PHASES[int(probs.argmax(dim=-1)[0].item())]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        action = _postprocess_sim_action(
            pred_action,
            processor,
            invert_gripper_action,
            binarize_gripper=bool(getattr(args, "binarize_gripper", True)),
        )
        return action, latency_ms

    results = _run_libero_rollout(
        cfg,
        args,
        mode=mode,
        predict_action_chunk=predict_action_chunk,
        result_extra={
            "strategy": "physics_onestep",
            "adapter_type": resolved_type,
            "use_physics": True,
            "phase_label_version": phase_label_version,
            "flow_sample_steps": flow_sample_steps,
            "adapter_ckpt": str(ckpt_path),
        },
    )
    # Record phase after rollouts (result_extra is snapshotted at call time).
    results["pred_phase"] = last_pred_phase
    results["phase_prob"] = last_phase_prob
    return results


def run_ours_physics_residual_sim(cfg: dict, args, *, adapter_type: str, mode: str) -> dict:
    """Physics residual over FastWAM.

    blend: a = (1-α)*a_FW + α*a_abs
    additive (Version C): a = a_FW + α*δ
    """
    residual_alpha = float(
        getattr(args, "residual_alpha", None)
        if getattr(args, "residual_alpha", None) is not None
        else (cfg.get("eval") or {}).get("residual_alpha", 0.1)
    )
    # α<=0 or tiny floor eps => pure FastWAM floor via official baseline (~96%).
    alpha_eff_floor_eps = float(
        getattr(args, "alpha_eff_floor_eps", None)
        if getattr(args, "alpha_eff_floor_eps", None) is not None
        else (cfg.get("eval") or {}).get("alpha_eff_floor_eps", 1e-4)
    )
    if residual_alpha <= max(0.0, alpha_eff_floor_eps):
        base = run_baseline_sim(
            cfg,
            max_tasks=int(args.max_tasks),
            num_trials=int(args.num_trials),
            device_override=args.device,
        )
        base["mode"] = mode
        base["strategy"] = "physics_residual_shortcircuit_baseline"
        base["residual_alpha"] = float(residual_alpha)
        base["residual_gate"] = "none"
        base["residual_mode"] = str(
            getattr(args, "residual_mode", None)
            or (cfg.get("eval") or {}).get("residual_mode")
            or "additive"
        )
        base["mean_alpha_eff"] = 0.0
        base["alpha_eff_floor_eps"] = alpha_eff_floor_eps
        return base

    cfg = _apply_fastwam_device(cfg, args.device)
    deps = _load_libero_eval_deps(cfg)
    invert_gripper_action = deps[4]  # deps: ..., image, invert_gripper, quat2axisangle, ...
    processor, _hydra_cfg = _build_fastwam_processor(cfg)

    cfg = dict(cfg)
    cfg["fastwam"] = dict(cfg.get("fastwam", {}))
    cfg["fastwam"]["load_text_encoder"] = True
    encoder = load_frozen_fastwam(cfg)

    device = cfg.get("fastwam", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    num_steps = _num_inference_steps(cfg, args)
    eval_cfg = cfg.get("eval") or {}
    residual_gate = str(
        getattr(args, "residual_gate", None)
        or eval_cfg.get("residual_gate")
        or "none"
    )
    residual_mode = str(
        getattr(args, "residual_mode", None)
        or eval_cfg.get("residual_mode")
        or "blend"
    ).lower()
    if residual_mode not in ("blend", "additive"):
        raise ValueError(f"--residual_mode must be blend|additive, got {residual_mode!r}")
    gate_confidence_thr = float(getattr(args, "gate_confidence_thr", 0.4))
    gate_disagree_thr = float(getattr(args, "gate_disagree_thr", 0.5))
    zero_alpha_on_uncertain = bool(
        getattr(args, "zero_alpha_on_uncertain", False)
        or eval_cfg.get("zero_alpha_on_uncertain", False)
    )
    if getattr(args, "adapter_phases", None):
        adapter_phases = _parse_adapter_phases(args)
    elif eval_cfg.get("adapter_phases"):
        class _Tmp:
            adapter_phases = eval_cfg.get("adapter_phases")
        adapter_phases = _parse_adapter_phases(_Tmp())
    else:
        adapter_phases = None
    flow_sample_steps = int(getattr(args, "flow_sample_steps", 10) or 10)

    if not args.adapter_ckpt:
        raise ValueError(f"--adapter_ckpt is required for --mode {mode}")
    ckpt_path = _resolve_path(args.adapter_ckpt)
    if ckpt_path is None or not ckpt_path.is_file():
        raise FileNotFoundError(f"adapter_ckpt not found: {args.adapter_ckpt}")

    override = getattr(args, "adapter_type", None) or adapter_type
    physics_model, resolved_type = _load_physics_model_for_eval(cfg, ckpt_path, override)
    phase_label_version = getattr(args, "phase_label_version", None) or cfg.get("physics", {}).get(
        "phase_label_version", "v1"
    )
    alpha_eff_hist: list[float] = []

    def predict_action_chunk(*, rgb, obs, task_desc, proprio, horizon):
        del obs
        obs_tensor = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        obs_tensor = obs_tensor * (2.0 / 255.0) - 1.0
        proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).float()

        batch_dit = {
            "obs_t": obs_tensor,
            "prompt": task_desc,
            "action_horizon": int(horizon),
            "proprio": proprio_tensor,
            "num_inference_steps": num_steps,
        }

        t0 = time.perf_counter()
        with inference_guard(), torch.no_grad():
            a_fastwam = encoder.infer_action_only(batch_dit)
            a_fw = a_fastwam.to(device)
            z_t = encoder.encode_obs_latent(obs_tensor)
            text_embed, _text_mask = encoder.encode_text_from_prompt(task_desc)
            text_embed = text_embed.to(device=device)
            out = physics_model.forward_inference(
                z_t.to(device),
                text_embed,
                state_t=proprio_tensor.to(device),
                num_flow_steps=flow_sample_steps,
            )
            delta = out.get("pred_action")
            if delta is None:
                raise RuntimeError("Physics model did not return pred_action")
            a_ad = delta.to(device)
            probs = out.get("phase_prob")
            if probs is None:
                probs = out.get("physics_probs")
            phase_name = None
            if probs is not None:
                phase_name = PHYSICS_PHASES[int(probs.argmax(dim=-1)[0].item())]
            # For disagreement gate in additive mode, compare reconstructed absolute.
            a_abs_for_gate = a_fw + a_ad if residual_mode == "additive" else a_ad
            alpha_eff = _effective_residual_alpha(
                residual_alpha,
                residual_gate=residual_gate,
                confidence=out.get("confidence"),
                a_fw=a_fw,
                a_adapter=a_abs_for_gate,
                gate_confidence_thr=gate_confidence_thr,
                gate_disagree_thr=gate_disagree_thr,
                adapter_phases=adapter_phases,
                phase_name=phase_name,
                zero_alpha_on_uncertain=zero_alpha_on_uncertain,
            )
            alpha_eff_hist.append(float(alpha_eff))
            if alpha_eff <= 0.0:
                pred_action = a_fw
            elif residual_mode == "additive":
                pred_action = a_fw + alpha_eff * a_ad
            else:
                pred_action = (1.0 - alpha_eff) * a_fw + alpha_eff * a_ad
        latency_ms = (time.perf_counter() - t0) * 1000.0
        action = _postprocess_sim_action(
            pred_action,
            processor,
            invert_gripper_action,
            binarize_gripper=bool(getattr(args, "binarize_gripper", True)),
        )
        return action, latency_ms

    results = _run_libero_rollout(
        cfg,
        args,
        mode=mode,
        predict_action_chunk=predict_action_chunk,
        result_extra={
            "strategy": "physics_residual",
            "residual_mode": residual_mode,
            "adapter_type": resolved_type,
            "use_physics": True,
            "phase_label_version": phase_label_version,
            "residual_alpha": residual_alpha,
            "residual_gate": residual_gate,
            "gate_confidence_thr": gate_confidence_thr,
            "gate_disagree_thr": gate_disagree_thr,
            "zero_alpha_on_uncertain": zero_alpha_on_uncertain,
            "adapter_phases": sorted(adapter_phases) if adapter_phases else None,
            "mean_alpha_eff": float(np.mean(alpha_eff_hist)) if alpha_eff_hist else None,
            "flow_sample_steps": flow_sample_steps,
            "adapter_ckpt": str(ckpt_path),
            "num_inference_steps": num_steps,
        },
    )
    # Rollout finishes after hist is filled; refresh mean for JSON.
    if alpha_eff_hist:
        results["mean_alpha_eff"] = float(np.mean(alpha_eff_hist))
        results["alpha_eff_n"] = len(alpha_eff_hist)
    return results


def run_ours_adapter_sim(cfg: dict, args) -> dict:
    """MLP ActionAdapter: z_t + text -> action (alias for ours_onestep_mlp)."""
    return run_ours_onestep_sim(cfg, args, adapter_type="mlp", mode="ours_adapter")


def run_ours_dit_sim(cfg: dict, args) -> dict:
    """FastWAM Action DiT: frozen infer_action() diffusion (matches idea2 design inference path)."""
    cfg = _apply_fastwam_device(cfg, args.device)
    deps = _load_libero_eval_deps(cfg)
    invert_gripper_action = deps[4]  # deps: ..., image, invert_gripper, quat2axisangle, ...
    processor, _hydra_cfg = _build_fastwam_processor(cfg)

    cfg = dict(cfg)
    cfg["fastwam"] = dict(cfg.get("fastwam", {}))
    cfg["fastwam"]["load_text_encoder"] = True
    encoder = load_frozen_fastwam(cfg)

    device = cfg.get("fastwam", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    num_steps = _num_inference_steps(cfg, args)

    def predict_action_chunk(*, rgb, obs, task_desc, proprio, horizon):
        del obs
        obs_tensor = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)
        obs_tensor = obs_tensor * (2.0 / 255.0) - 1.0
        proprio_tensor = torch.from_numpy(proprio).unsqueeze(0).float()

        batch = {
            "obs_t": obs_tensor,
            "prompt": task_desc,
            "action_horizon": int(horizon),
            "proprio": proprio_tensor,
            "num_inference_steps": num_steps,
        }

        t0 = time.perf_counter()
        with inference_guard(), torch.no_grad():
            pred_action = encoder.infer_action_only(batch)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        action = _postprocess_sim_action(
            pred_action,
            processor,
            invert_gripper_action,
            binarize_gripper=bool(getattr(args, "binarize_gripper", True)),
        )
        return action, latency_ms

    return _run_libero_rollout(
        cfg,
        args,
        mode="ours_dit",
        predict_action_chunk=predict_action_chunk,
        result_extra={
            "strategy": "fastwam_action_dit",
            "num_inference_steps": num_steps,
            "fastwam_checkpoint": str(cfg.get("official_fastwam_checkpoint", "")),
        },
    )


def run_ours_residual_sim(cfg: dict, args) -> dict:
    """Residual MLP adapter (alias for ours_residual_mlp)."""
    return run_ours_residual_variant_sim(cfg, args, adapter_type="mlp", mode="ours_residual")


def run_offline_eval(cfg: dict, args) -> dict:
    cfg = _apply_fastwam_device(cfg, args.device)
    device = cfg.get("fastwam", {}).get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    encoder = load_frozen_fastwam(cfg)
    results = {"mode": args.mode}
    cache_dir = _resolve_path(args.cache_dir or cfg["cache"]["output_dir"])
    results.update(measure_infer_latency(encoder, device, cache_dir=cache_dir))

    if args.latent_verification:
        if cache_dir is None:
            raise ValueError("cache_dir required for latent_verification")
        ckpt = _resolve_path(args.heads_ckpt)
        results.update(latent_verification(cfg, cache_dir, ckpt, device))

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/world2wam_libero_spatial_h10.yaml")
    parser.add_argument(
        "--mode",
        choices=[
            "baseline",
            "ours_adapter",
            "ours_dit",
            "ours_residual",
            "ours_onestep_mlp",
            "ours_onestep_light_dit",
            "ours_onestep_flow_dit",
            "ours_residual_mlp",
            "ours_residual_light_dit",
            "ours_residual_flow_dit",
            "ours_onestep_physics_mlp",
            "ours_onestep_physics_light_dit",
            "ours_onestep_physics_flow_dit",
            "ours_residual_physics_mlp",
            "ours_residual_physics_light_dit",
            "ours_residual_physics_flow_dit",
            "ours_residual_physics_flow_dit_vc",
            "offline",
        ],
        default="baseline",
    )
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--heads_ckpt", default=None)
    parser.add_argument("--adapter_ckpt", default=None)
    parser.add_argument("--latent_verification", action="store_true")
    parser.add_argument("--max_tasks", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=30,
        help="Dummy env steps before control (official FastWAM num_steps_wait=30)",
    )
    parser.add_argument(
        "--replan_steps",
        type=int,
        default=10,
        help="Execute first K actions of each chunk then replan (official FastWAM replan_steps=10)",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=None,
        help="Diffusion steps for --mode ours_dit / ours_residual (default: config eval.num_inference_steps or 10)",
    )
    parser.add_argument(
        "--binarize_gripper",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match official FastWAM binarize_gripper (default: true)",
    )
    parser.add_argument("--residual_alpha", type=float, default=None)
    parser.add_argument(
        "--alpha_eff_floor_eps",
        type=float,
        default=None,
        help="If residual_alpha <= eps, shortcircuit to official baseline (default 1e-4).",
    )
    parser.add_argument(
        "--residual_mode",
        choices=["blend", "additive"],
        default=None,
        help="blend: (1-α)a_FW+α a_abs; additive (Version C): a_FW+α·δ",
    )
    parser.add_argument(
        "--residual_gate",
        choices=["none", "confidence_soft", "confidence_hard", "disagreement"],
        default=None,
        help="Lean FastWAM when unsure/disagreeing: shrink effective residual alpha.",
    )
    parser.add_argument(
        "--gate_confidence_thr",
        type=float,
        default=0.4,
        help="For confidence_hard: use adapter only if router confidence >= thr.",
    )
    parser.add_argument(
        "--gate_disagree_thr",
        type=float,
        default=0.5,
        help="For disagreement: α_eff = α * max(0, 1 - mean|a_ad-a_fw|/thr).",
    )
    parser.add_argument(
        "--zero_alpha_on_uncertain",
        action="store_true",
        help="Force α_eff=0 when predicted phase is 'uncertain' (physics residual).",
    )
    parser.add_argument(
        "--adapter_phases",
        default=None,
        help="Comma-separated phases where adapter is allowed; else α_eff=0. e.g. grasp,place,contact",
    )
    parser.add_argument(
        "--task_ids",
        default=None,
        help="Comma-separated task ids to eval (overrides --max_tasks), e.g. 4,7,9",
    )
    parser.add_argument("--adapter_type", choices=["mlp", "light_dit", "flow_dit"], default=None)
    parser.add_argument("--flow_sample_steps", type=int, default=10)
    parser.add_argument("--phase_label_version", default="v1")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cfg = load_config(WORKSPACE / args.config)
    set_seed(int(cfg.get("train", {}).get("seed", 42)))
    eval_cfg = cfg.get("eval") or {}
    # Fill residual eval defaults from config when CLI omitted.
    if args.residual_alpha is None:
        args.residual_alpha = float(eval_cfg.get("residual_alpha", 0.1))
    if args.residual_mode is None:
        args.residual_mode = str(eval_cfg.get("residual_mode", "blend"))
    if args.residual_gate is None:
        args.residual_gate = str(eval_cfg.get("residual_gate", "none"))
    if not args.zero_alpha_on_uncertain and eval_cfg.get("zero_alpha_on_uncertain"):
        args.zero_alpha_on_uncertain = True
    if args.adapter_phases is None and eval_cfg.get("adapter_phases"):
        args.adapter_phases = eval_cfg.get("adapter_phases")
    # Align custom loop knobs with official FastWAM when config provides them.
    if eval_cfg.get("warmup_steps") is not None and args.warmup_steps == 30:
        args.warmup_steps = int(eval_cfg["warmup_steps"])
    if eval_cfg.get("replan_steps") is not None and args.replan_steps == 10:
        args.replan_steps = int(eval_cfg["replan_steps"])
    if "binarize_gripper" in eval_cfg and args.binarize_gripper is True:
        args.binarize_gripper = bool(eval_cfg["binarize_gripper"])
    if args.num_inference_steps is None and eval_cfg.get("num_inference_steps") is not None:
        args.num_inference_steps = int(eval_cfg["num_inference_steps"])

    if args.mode == "baseline":
        results = run_baseline_sim(
            cfg,
            max_tasks=args.max_tasks,
            num_trials=args.num_trials,
            device_override=args.device,
        )
    elif args.mode == "offline":
        results = run_offline_eval(cfg, args)
    elif args.mode == "ours_adapter":
        results = run_ours_adapter_sim(cfg, args)
    elif args.mode == "ours_onestep_mlp":
        results = run_ours_onestep_sim(cfg, args, adapter_type="mlp", mode="ours_onestep_mlp")
    elif args.mode == "ours_onestep_light_dit":
        results = run_ours_onestep_sim(cfg, args, adapter_type="light_dit", mode="ours_onestep_light_dit")
    elif args.mode == "ours_onestep_flow_dit":
        results = run_ours_onestep_sim(cfg, args, adapter_type="flow_dit", mode="ours_onestep_flow_dit")
    elif args.mode == "ours_dit":
        results = run_ours_dit_sim(cfg, args)
    elif args.mode == "ours_residual":
        results = run_ours_residual_sim(cfg, args)
    elif args.mode == "ours_residual_mlp":
        results = run_ours_residual_variant_sim(cfg, args, adapter_type="mlp", mode="ours_residual_mlp")
    elif args.mode == "ours_residual_light_dit":
        results = run_ours_residual_variant_sim(cfg, args, adapter_type="light_dit", mode="ours_residual_light_dit")
    elif args.mode == "ours_residual_flow_dit":
        results = run_ours_residual_variant_sim(cfg, args, adapter_type="flow_dit", mode="ours_residual_flow_dit")
    elif args.mode == "ours_onestep_physics_mlp":
        results = run_ours_physics_onestep_sim(cfg, args, adapter_type="mlp", mode="ours_onestep_physics_mlp")
    elif args.mode == "ours_onestep_physics_light_dit":
        results = run_ours_physics_onestep_sim(cfg, args, adapter_type="light_dit", mode="ours_onestep_physics_light_dit")
    elif args.mode == "ours_onestep_physics_flow_dit":
        results = run_ours_physics_onestep_sim(cfg, args, adapter_type="flow_dit", mode="ours_onestep_physics_flow_dit")
    elif args.mode == "ours_residual_physics_mlp":
        results = run_ours_physics_residual_sim(cfg, args, adapter_type="mlp", mode="ours_residual_physics_mlp")
    elif args.mode == "ours_residual_physics_light_dit":
        results = run_ours_physics_residual_sim(cfg, args, adapter_type="light_dit", mode="ours_residual_physics_light_dit")
    elif args.mode == "ours_residual_physics_flow_dit":
        results = run_ours_physics_residual_sim(cfg, args, adapter_type="flow_dit", mode="ours_residual_physics_flow_dit")
    elif args.mode == "ours_residual_physics_flow_dit_vc":
        # Version C primary: additive residual + physics gates from VC config defaults.
        args.residual_mode = "additive"
        if args.residual_gate in (None, "none") and eval_cfg.get("residual_gate"):
            args.residual_gate = str(eval_cfg.get("residual_gate"))
        results = run_ours_physics_residual_sim(
            cfg, args, adapter_type="flow_dit", mode="ours_residual_physics_flow_dit_vc"
        )
    else:
        raise ValueError(args.mode)

    out_path = _resolve_path(args.output) if args.output else _resolve_path(
        Path(cfg.get("train", {}).get("output_dir", "experiments")) / f"eval_{args.mode}.json"
    )
    assert out_path is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

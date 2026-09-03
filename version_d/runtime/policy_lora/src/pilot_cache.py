"""
pilot_cache.py — pilot 离线 cache 的生成 / 加载（遵循学长 guide 的"冻结串行"方案）。

Cache 内容（冻结 base FastWAM 后离线导出，不重训 base）：
  a0   : [N, T, D]   冻结 base FastWAM 输出 (forward_action_only)
  c    : [N, cond_dim] 条件（action hidden 表征，复用 FastWAM vision tower 出口）
  a_gt : [N, T, D]   目标"专家"动作
            - Round 1 (pilot): 用 GT（Flow 作为对 GT 的 corrector）
            - 实验③ (联合): 可换学长 LoRA 输出做 teacher（改 a_gt 来源即可）

依赖边界：
  - generate_cache() 需要 GPU + 重依赖（habitat / transformers / senior wrapper），
    在服务器腾出卡后调用。
  - load_cache() / split_cache() / train_flow_pilot / compare_pilot 只吃 cache，
    纯 CPU、仅依赖 torch，满足"无卡也能跑管线"的初衷。
"""
from __future__ import annotations

import torch

CACHE_KEYS = ("a0", "c", "a_gt")


def save_cache(path: str, a0: torch.Tensor, c: torch.Tensor, a_gt: torch.Tensor, meta: dict | None = None) -> None:
    if a0.dim() != 3 or c.dim() != 2 or a_gt.dim() != 3:
        raise ValueError("shape mismatch: a0/a_gt [N,T,D], c [N,cond_dim]")
    if a0.shape[0] != c.shape[0] or a0.shape[0] != a_gt.shape[0]:
        raise ValueError("batch size must match across a0/c/a_gt")
    torch.save(
        {"a0": a0.cpu(), "c": c.cpu(), "a_gt": a_gt.cpu(), "meta": meta or {}},
        path,
    )


def load_cache(path: str) -> dict:
    d = torch.load(path, map_location="cpu", weights_only=True)
    for k in CACHE_KEYS:
        if k not in d:
            raise KeyError(f"cache {path} missing key '{k}'")
    return d


def split_cache(d: dict, train_frac: float = 0.8, seed: int = 0) -> tuple[dict, dict]:
    n = d["a0"].shape[0]
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g)
    nt = max(1, int(n * train_frac))
    tr, va = idx[:nt], idx[nt:]
    return ({k: d[k][tr] for k in CACHE_KEYS}, {k: d[k][va] for k in CACHE_KEYS})


def generate_cache(
    wrapper,
    dataloader,
    cond_fn,
    out_path: str,
    device: str = "cuda",
    max_samples: int = 2000,
) -> str:
    """
    GPU 生成（服务器腾卡后调用）。
      wrapper   : 冻结的 FastWAMWrapper（eval 模式）
      dataloader: 返回 batch，batch['action'] 为 GT 动作 [B,T,D]
      cond_fn   : callable(batch) -> [B, cond_dim]，抽取 action hidden 条件
                  建议复用 FastWAM vision tower 出口；默认回退见 _default_cond_fn。
    导出到 out_path，返回路径。
    """
    a0s, c0s, agts = [], [], []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            b = batch["action"].shape[0]
            if i * b >= max_samples:
                break
            out = wrapper.forward_action_only(batch)
            a0 = out["pred_action"].to(device)
            c = cond_fn(batch).to(device)
            a_gt = batch["action"].to(device)
            a0s.append(a0.cpu())
            c0s.append(c.cpu())
            agts.append(a_gt.cpu())
    save_cache(
        out_path,
        torch.cat(a0s),
        torch.cat(c0s),
        torch.cat(agts),
        meta={"source": "version_d_fixed_ckpt", "device": device, "n": sum(x.shape[0] for x in a0s)},
    )
    return out_path


def _default_cond_fn(wrapper, batch: dict):
    """默认条件抽取：优先 wrapper.get_flow_cond，否则 batch['flow_cond']。
    不偷偷改动 base 推理图；取不到就显式报错由调用方处理。"""
    if hasattr(wrapper, "get_flow_cond"):
        return wrapper.get_flow_cond(batch)
    if "flow_cond" in batch:
        return batch["flow_cond"]
    raise ValueError("cond_fn not provided and no flow_cond available; pass cond_fn explicitly.")

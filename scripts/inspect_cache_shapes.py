#!/usr/bin/env python3
"""Inspect random cache .pt files for keys and tensor shapes."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import torch

WORKSPACE = Path(__file__).resolve().parents[2]
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

from minimal_world2wam.data.latent_cache_dataset import list_cache_files, load_meta


REQUIRED_KEYS = ("z_t", "z_tH", "text_embed", "action_chunk")


def inspect(cache_dir: Path, num_samples: int, seed: int = 42) -> dict:
    files = list_cache_files(cache_dir)
    meta = load_meta(cache_dir)
    rng = random.Random(seed)
    chosen = rng.sample(files, min(num_samples, len(files)))

    shape_records: dict[str, list[tuple]] = {k: [] for k in REQUIRED_KEYS}
    all_keys: Counter[str] = Counter()

    for fp in chosen:
        payload = torch.load(fp, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError(f"{fp.name}: expected dict payload")
        all_keys.update(payload.keys())
        for key in REQUIRED_KEYS:
            if key not in payload:
                raise KeyError(f"{fp.name}: missing key {key}")
            t = payload[key]
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"{fp.name}: {key} is not a tensor")
            shape_records[key].append(tuple(t.shape))

    summary: dict = {
        "cache_dir": str(cache_dir),
        "num_inspected": len(chosen),
        "meta": meta,
        "all_keys_seen": sorted(all_keys.keys()),
        "shapes": {},
        "consistent": True,
    }

    for key, shapes in shape_records.items():
        unique = list(dict.fromkeys(shapes))
        summary["shapes"][key] = {"unique_shapes": unique, "count": len(shapes)}
        if len(unique) != 1:
            summary["consistent"] = False
            print(f"WARNING: inconsistent shapes for {key}: {unique}")

    print(f"Cache: {cache_dir}")
    print(f"Inspected {len(chosen)} / {len(files)} files")
    print(f"Meta: {json.dumps(meta, indent=2)}")
    for key in REQUIRED_KEYS:
        info = summary["shapes"][key]
        print(f"  {key}: {info['unique_shapes']}")
    print(f"All keys seen: {summary['all_keys_seen']}")
    print(f"Consistent: {summary['consistent']}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_absolute():
        cache_dir = (WORKSPACE / cache_dir).resolve()

    summary = inspect(cache_dir, args.num_samples, seed=args.seed)
    if args.output:
        out = Path(args.output)
        if not out.is_absolute():
            out = (WORKSPACE / out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()

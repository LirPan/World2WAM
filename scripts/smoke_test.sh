#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${WORKSPACE}"
source /DATA/disk0/jianhua/miniconda3/etc/profile.d/conda.sh
conda activate world2wam
export PYTHONPATH="${WORKSPACE}:${PYTHONPATH:-}"

echo "== config load =="
python -c "
from minimal_world2wam.utils.config import load_config
cfg = load_config('configs/world2wam_libero_spatial_h10.yaml')
print('fastwam_root', cfg['fastwam_root'])
print('horizon', cfg['horizon'])
"

echo "== transition dataset (5 samples) =="
python -c "
from minimal_world2wam.utils.config import load_config
from minimal_world2wam.data.libero_transition_dataset import LiberoTransitionDataset, build_fastwam_dataset
cfg = load_config('configs/world2wam_libero_spatial_h10.yaml')
base, _ = build_fastwam_dataset(cfg)
ds = LiberoTransitionDataset(base, horizon=cfg['horizon'], max_samples=5)
s = ds[0]
print('len', len(ds))
print('obs_t', tuple(s['obs_t'].shape))
print('obs_tH', tuple(s['obs_tH'].shape))
print('action_chunk', tuple(s['action_chunk'].shape))
print('context', tuple(s['context'].shape))
"

echo "SMOKE OK"

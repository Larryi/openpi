# Weighted LeRobot v3 mixtures

This repository supports virtual mixtures of multiple LeRobot v3 datasets without copying parquet files,
videos, or converting dataset versions.

## Design

Each `LeRobotDatasetConfig` describes one source:

- `repo_id`: Hub dataset id, or an arbitrary stable id for a local dataset.
- `root`: exact local dataset directory; leave unset to use the Hub/cache.
- `episodes`: optional episode-index subset.
- `weight`: desired source probability in training batches.
- `name`: unique label used in logs.

OpenPI opens every source with the regular LeRobot API:

```python
LeRobotDataset(
    source.repo_id,
    root=source.root,
    episodes=list(source.episodes) if source.episodes is not None else None,
    delta_timestamps=...,
)
```

LeRobot therefore remains responsible for video decoding, action chunks, and episode-tail padding. The
OpenPI `WeightedLeRobotDataset` only maps concatenated indices to the underlying datasets.

For a source with requested probability `p_s` and `N_s` frames, each frame receives sampler weight:

```text
w_i = p_s / N_s
```

Consequently, the sum of frame weights for that source is `p_s`, regardless of dataset size. Sampling uses
replacement, which prevents a large historical dataset from dominating a smaller target-domain dataset.

The same weighted sampler is used by `compute_norm_stats.py`; normalization and training therefore see the
same source distribution.

## Kuavo configurations

Two configurations are provided:

- `pi05_kuavo_task1_mixed`
- `pi05_kuavo_task2_mixed`

Both currently use:

```text
Beijing main       0.55
Beijing slave      0.20
Suzhou repaired    0.25
```

The Beijing episode subsets were recovered from `header_export_manifest.json` and verified after idle-action
trimming:

```text
Task1 main  0..103      slave 104..219
Task2 main  0..101      slave 102..207
```

Run the lightweight loader checks:

```bash
export UV_DEFAULT_INDEX=https://mirrors.bfsu.edu.cn/pypi/web/simple

uv run scripts/smoke_test_kuavo_mixture.py \
  --config-name pi05_kuavo_task1_mixed

uv run scripts/smoke_test_kuavo_mixture.py \
  --config-name pi05_kuavo_task2_mixed
```

Then compute full mixed normalization statistics before a fresh training run:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_kuavo_task1_mixed \
  --batch-size 32 \
  --num-workers 8
```

The output asset ids are `kuavo_task1_sz_bj_mixed` and `kuavo_task2_sz_bj_mixed`.

## Continuing the Task1 45k checkpoint

Mixed statistics are appropriate for a fresh run from the Physical Intelligence base parameters. A model
already trained for 45k steps with the old Task1 statistics should not silently switch normalization.

For a continuation experiment:

1. Keep the original Task1 normalization assets (`kuavo_task1`).
2. Load the 45k model parameters, or resume the complete Orbax train state when available.
3. Use the mixed dataset sampler.
4. Start a new experiment name and a low-LR tail schedule.
5. Compare old and mixed q01/q99 before deciding whether a deliberate normalization migration is worthwhile.

If Hugging Face contains only `params/`, this is weight initialization with a new optimizer, not a true
optimizer-state resume.

## Porting to LingBot or Diffusion Policy

The portable pieces are:

1. A source descriptor containing `root/repo_id/episodes/weight`.
2. One normal `LeRobotDataset` instance per source.
3. A concatenating dataset that preserves source lengths.
4. Per-frame weights `source_weight / source_length`.
5. A replacement sampler passed to `DataLoader`; `shuffle` must be disabled when a sampler is present.
6. The same sampler distribution for normalization-stat computation.

For single-process training:

```python
sampler = torch.utils.data.WeightedRandomSampler(
    per_frame_weights,
    num_samples=len(mixture),
    replacement=True,
    generator=generator,
)
loader = DataLoader(mixture, sampler=sampler, shuffle=False, ...)
```

For PyTorch DDP, do not give every rank an identically seeded `WeightedRandomSampler`, because ranks would
receive duplicate index streams. Each rank should independently sample with:

```text
seed = base_seed + epoch * world_size + rank
num_samples_per_rank = ceil(samples_per_virtual_epoch / world_size)
```

The sampler should expose `set_epoch(epoch)` so resumed DDP training reproduces the expected sequence. Since
sampling is with replacement, independent rank streams are valid and no physical dataset partition is
required.

For Diffusion Policy, preserve action semantics and compute normalization after the robot-specific action
transform. For LingBot, preserve the existing image/state/action transforms and replace only dataset
selection and sampling.

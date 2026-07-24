#!/usr/bin/env python3
"""Open-loop diagnostics for a JAX Pi0.5 policy on local Kuavo LeRobot v3 data."""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime
import json
import logging
import pathlib
import time

import jax
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
from tqdm import tqdm

from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

DEFAULT_CHECKPOINT = pathlib.Path("/mnt/pqssd/pretrained/pi05_local_jax")


@dataclasses.dataclass
class EvaluationStats:
    sum_abs: np.ndarray
    sum_sq: np.ndarray
    count_horizon: np.ndarray
    range_violations: int = 0
    range_values: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a local OpenPI JAX Pi0.5 checkpoint and compare action chunks with Kuavo LeRobot v3 GT."
    )
    parser.add_argument("--config-name", default="pi05_kuavo", choices=("pi05_kuavo", "pi05_kuavo_task2"))
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint root containing params/, or the params/ directory itself.",
    )
    parser.add_argument("--dataset-root", type=pathlib.Path, default=None)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--episodes", type=int, nargs="*", default=[0])
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic flow-matching noise.")
    parser.add_argument("--num-steps", type=int, default=10, help="Flow-matching integration steps.")
    parser.add_argument(
        "--video-backend",
        choices=("torchcodec", "pyav", "video_reader"),
        default="torchcodec",
        help="torchcodec is recommended; torchvision builds without VideoReader cannot use pyav/video_reader.",
    )
    parser.add_argument("--output-dir", type=pathlib.Path, default=None)
    parser.add_argument("--load-only", action="store_true", help="Load the model and transforms, then exit.")
    parser.add_argument(
        "--require-cuda", action="store_true", help="Fail instead of silently running inference on CPU."
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def checkpoint_root(path: pathlib.Path) -> pathlib.Path:
    path = path.expanduser().resolve()
    if path.name == "params" and (path / "_METADATA").is_file():
        path = path.parent
    params = path / "params"
    if not (params / "_METADATA").is_file():
        raise FileNotFoundError(f"Expected an Orbax checkpoint at {params}")
    return path


def scalar_int(value) -> int:
    return int(np.asarray(value).item())


def make_observation(sample: dict) -> dict:
    observation = {
        key: np.asarray(value)
        for key, value in sample.items()
        if key.startswith("observation.images.") or key == "observation.state"
    }
    task = sample.get("task")
    if task is None:
        raise KeyError("LeRobot v3 sample does not contain a task prompt")
    observation["prompt"] = task
    return observation


def output_directory(args: argparse.Namespace) -> pathlib.Path:
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    timestamp = datetime.datetime.now(tz=datetime.UTC).astimezone().strftime("%Y%m%d_%H%M%S")
    return (pathlib.Path("outputs/open_loop") / args.config_name / timestamp).resolve()


def optional_float_list(values: np.ndarray, valid: np.ndarray) -> list[float | None]:
    return [float(value) if is_valid else None for value, is_valid in zip(values, valid, strict=True)]


def write_outputs(output_dir: pathlib.Path, summary: dict, rows: list[dict], stats: EvaluationStats) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    with (output_dir / "first_action_errors.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(
        output_dir / "error_stats.npz",
        sum_abs=stats.sum_abs,
        sum_sq=stats.sum_sq,
        count_horizon=stats.count_horizon,
    )


def write_plots(output_dir: pathlib.Path, horizon_mae: list[float | None], dim_mae: list[float]) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError as exc:
        logging.warning("Skipping open-loop plots because matplotlib is unavailable: %s", exc)
        return

    horizon_values = np.asarray([np.nan if value is None else value for value in horizon_mae])
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.plot(np.arange(len(horizon_values)), horizon_values)
    axis.set(xlabel="horizon step", ylabel="MAE", title="Kuavo Pi0.5 open-loop horizon MAE")
    figure.tight_layout()
    figure.savefig(output_dir / "horizon_mae.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(np.arange(len(dim_mae)), dim_mae)
    axis.set(xlabel="action dimension", ylabel="MAE", title="Kuavo Pi0.5 open-loop per-dimension MAE")
    figure.tight_layout()
    figure.savefig(output_dir / "dim_mae.png", dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.stride <= 0 or args.max_samples <= 0 or args.num_steps <= 0:
        raise ValueError("stride, max-samples, and num-steps must all be positive")

    train_config = _config.get_config(args.config_name)
    data_factory = train_config.data
    dataset_root = (args.dataset_root or pathlib.Path(data_factory.root or "")).expanduser().resolve()
    repo_id = args.repo_id or data_factory.repo_id
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Kuavo dataset root does not exist: {dataset_root}")
    if repo_id is None:
        raise ValueError("repo-id is required")

    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    if data_config.norm_stats is None:
        raise FileNotFoundError(f"No Kuavo norm stats for {args.config_name}; run scripts/compute_norm_stats.py first")

    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    _data_loader.validate_lerobot_metadata(metadata, data_config)
    checkpoint = checkpoint_root(args.checkpoint)
    devices = jax.devices()
    if args.require_cuda and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"CUDA was required, but JAX only found these devices: {devices}")
    logging.info("JAX devices: %s", devices)
    logging.info("Loading %s with config %s", checkpoint, args.config_name)
    load_started = time.perf_counter()
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint,
        norm_stats=data_config.norm_stats,
        sample_kwargs={"num_steps": args.num_steps},
    )
    load_seconds = time.perf_counter() - load_started
    print(f"PASS: loaded JAX Pi0.5 policy from {checkpoint} in {load_seconds:.2f}s")
    print(f"devices: {[str(device) for device in devices]}")
    print(f"config: {args.config_name}; robot action dim: {data_factory.action_dim}")
    if args.load_only:
        return

    action_horizon = train_config.model.action_horizon
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=args.episodes or None,
        delta_timestamps={"action": [step / metadata.fps for step in range(action_horizon)]},
        video_backend=args.video_backend,
    )

    action_dim = data_factory.action_dim
    stats = EvaluationStats(
        sum_abs=np.zeros((action_horizon, action_dim), dtype=np.float64),
        sum_sq=np.zeros((action_horizon, action_dim), dtype=np.float64),
        count_horizon=np.zeros(action_horizon, dtype=np.int64),
    )
    action_stats = metadata.stats.get("action", {}) if metadata.stats else {}
    action_min = np.asarray(action_stats.get("min"), dtype=np.float32) if "min" in action_stats else None
    action_max = np.asarray(action_stats.get("max"), dtype=np.float32) if "max" in action_stats else None
    delta_mask = np.asarray(data_factory.delta_action_mask, dtype=bool)
    rng = np.random.default_rng(args.seed)
    rows: list[dict] = []
    inference_seconds: list[float] = []
    nonfinite_values = 0
    predicted_values = 0

    progress = tqdm(range(len(dataset)), desc=f"open-loop {args.config_name}")
    for dataset_index in progress:
        # Check the parquet-backed scalar before __getitem__ decodes camera videos. This makes sparse
        # evaluation proportional to the number of selected observations instead of the stride.
        frame_index = scalar_int(dataset.hf_dataset[dataset_index]["frame_index"])
        if frame_index % args.stride != 0:
            continue
        sample = dataset[dataset_index]

        noise = rng.standard_normal((action_horizon, train_config.model.action_dim), dtype=np.float32)
        infer_started = time.perf_counter()
        result = policy.infer(make_observation(sample), noise=noise)
        inference_seconds.append(time.perf_counter() - infer_started)

        predicted = np.asarray(result["actions"], dtype=np.float32).reshape(-1, action_dim)
        target = np.asarray(sample["action"], dtype=np.float32).reshape(-1, action_dim)
        horizon = min(action_horizon, len(predicted), len(target))
        valid = np.ones(horizon, dtype=bool)
        if "action_is_pad" in sample:
            valid &= ~np.asarray(sample["action_is_pad"], dtype=bool)[:horizon]
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:
            continue

        predicted = predicted[:horizon][valid]
        target = target[:horizon][valid]
        finite = np.isfinite(predicted)
        nonfinite_values += int((~finite).sum())
        predicted_values += int(predicted.size)
        if not finite.all():
            continue

        error = predicted - target
        stats.sum_abs[valid_indices] += np.abs(error)
        stats.sum_sq[valid_indices] += np.square(error)
        stats.count_horizon[valid_indices] += 1
        if action_min is not None and action_max is not None:
            stats.range_violations += int(((predicted < action_min) | (predicted > action_max)).sum())
            stats.range_values += int(predicted.size)

        first_error = error[0]
        row = {
            "dataset_index": dataset_index,
            "episode_index": scalar_int(sample["episode_index"]),
            "frame_index": frame_index,
            "prompt": str(sample["task"]),
            "valid_horizon": int(valid_indices.size),
            "infer_ms": float(result.get("policy_timing", {}).get("infer_ms", inference_seconds[-1] * 1000)),
            "first_action_mae": float(np.abs(first_error).mean()),
            "first_action_rmse": float(np.sqrt(np.square(first_error).mean())),
        }
        for dim in range(action_dim):
            row[f"pred_{dim}"] = float(predicted[0, dim])
            row[f"gt_{dim}"] = float(target[0, dim])
            row[f"err_{dim}"] = float(first_error[dim])
        rows.append(row)
        progress.set_postfix(samples=len(rows))
        if len(rows) >= args.max_samples:
            break

    if not rows:
        raise RuntimeError("No finite samples were evaluated; check episodes, stride, and dataset contents")

    observed_horizon = stats.count_horizon > 0
    mae = np.divide(
        stats.sum_abs,
        stats.count_horizon[:, None],
        out=np.full_like(stats.sum_abs, np.nan),
        where=observed_horizon[:, None],
    )
    mse = np.divide(
        stats.sum_sq,
        stats.count_horizon[:, None],
        out=np.full_like(stats.sum_sq, np.nan),
        where=observed_horizon[:, None],
    )
    inference_array = np.asarray(inference_seconds, dtype=np.float64)
    summary = {
        "checkpoint": str(checkpoint),
        "config_name": args.config_name,
        "dataset_root": str(dataset_root),
        "repo_id": repo_id,
        "episodes": args.episodes,
        "stride": args.stride,
        "samples": len(rows),
        "action_horizon": action_horizon,
        "action_dim": action_dim,
        "model_action_dim": train_config.model.action_dim,
        "load_seconds": load_seconds,
        "first_inference_seconds": float(inference_array[0]),
        "steady_inference_seconds_mean": (float(inference_array[1:].mean()) if len(inference_array) > 1 else None),
        "steady_inference_seconds_p95": (
            float(np.percentile(inference_array[1:], 95)) if len(inference_array) > 1 else None
        ),
        "overall_mae": float(np.nanmean(mae)),
        "overall_rmse": float(np.sqrt(np.nanmean(mse))),
        "first_action_mae": float(np.nanmean(mae[0])),
        "horizon_mae": optional_float_list(
            np.divide(
                stats.sum_abs.sum(axis=1),
                stats.count_horizon * action_dim,
                out=np.zeros(action_horizon, dtype=np.float64),
                where=observed_horizon,
            ),
            observed_horizon,
        ),
        "dim_mae": np.nanmean(mae, axis=0).tolist(),
        "joint_mae": float(np.nanmean(mae[:, delta_mask])),
        "gripper_mae": float(np.nanmean(mae[:, ~delta_mask])),
        "nonfinite_rate": float(nonfinite_values / predicted_values) if predicted_values else None,
        "range_violation_rate": (float(stats.range_violations / stats.range_values) if stats.range_values else None),
        "jax_devices": [str(device) for device in jax.devices()],
        "note": "This is open-loop imitation diagnostics; low MAE is not expected from an unfine-tuned base model.",
    }
    output_dir = output_directory(args)
    write_outputs(output_dir, summary, rows, stats)
    if not args.no_plots:
        write_plots(output_dir, summary["horizon_mae"], summary["dim_mae"])
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote open-loop diagnostics to: {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

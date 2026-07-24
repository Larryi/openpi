#!/usr/bin/env python3
"""Load an SO101 JAX checkpoint and infer one real LeRobot v3 observation without hardware."""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import jax
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np

from openpi.policies import policy_config
from openpi.training import config as _config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_so101_60")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/mnt/pqssd/so101/datasets/merged_grab_blue_pen_60")
    )
    parser.add_argument("--repo-id", default="so101_grab_blue_pen_60")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--rtc", action="store_true", help="Also compile and run one denoising-level RTC request.")
    parser.add_argument("--rtc-inference-delay", type=int, default=5)
    parser.add_argument("--rtc-execution-horizon", type=int, default=20)
    parser.add_argument("--rtc-max-guidance-weight", type=float, default=10.0)
    parser.add_argument("--rtc-runs", type=int, default=2, help="RTC calls to run; the first includes compilation.")
    args = parser.parse_args()

    devices = jax.devices()
    if args.require_cuda and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError(f"CUDA required, but JAX found {devices}")
    config = _config.get_config(args.config_name)
    started = time.perf_counter()
    policy = policy_config.create_trained_policy(config, args.checkpoint.expanduser().resolve())
    print(f"Loaded checkpoint in {time.perf_counter() - started:.2f}s on {devices}")

    frame = LeRobotDataset(args.repo_id, root=args.dataset_root.expanduser().resolve())[args.frame]
    observation = {
        key: np.asarray(value)
        for key, value in frame.items()
        if key in ("observation.images.front", "observation.images.wrist", "observation.state")
    }
    observation["prompt"] = str(frame["task"])
    started = time.perf_counter()
    result = policy.infer(observation)
    actions = np.asarray(result["actions"], dtype=np.float32)
    print(f"Inference completed in {time.perf_counter() - started:.2f}s")
    print("prompt:", observation["prompt"])
    print("state:", observation["observation.state"])
    print("actions shape:", actions.shape)
    print("actions min/max:", actions.min(axis=0), actions.max(axis=0))
    assert actions.shape == (50, 6), actions.shape
    assert np.isfinite(actions).all()
    if args.rtc:
        rtc_observation = {
            **observation,
            "rtc": {
                "prev_actions": actions[: args.rtc_execution_horizon],
                "inference_delay": args.rtc_inference_delay,
                "execution_horizon": args.rtc_execution_horizon,
                "max_guidance_weight": args.rtc_max_guidance_weight,
            },
        }
        for run in range(args.rtc_runs):
            started = time.perf_counter()
            rtc_result = policy.infer(rtc_observation)
            rtc_actions = np.asarray(rtc_result["actions"], dtype=np.float32)
            print(f"RTC inference {run + 1}/{args.rtc_runs} completed in {time.perf_counter() - started:.2f}s")
            print("RTC actions shape:", rtc_actions.shape)
            print("RTC actions min/max:", rtc_actions.min(axis=0), rtc_actions.max(axis=0))
            assert rtc_actions.shape == (50, 6), rtc_actions.shape
            assert np.isfinite(rtc_actions).all()
    print("PASS: SO101 checkpoint offline inference")


if __name__ == "__main__":
    main()

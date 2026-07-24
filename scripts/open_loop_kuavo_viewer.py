#!/usr/bin/env python3
"""Interactive Streamlit viewer for OpenPI JAX Pi0.5 on Kuavo LeRobot v3 datasets."""

from __future__ import annotations

import pathlib
import time

import jax
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageEnhance
import streamlit as st

from openpi.policies import policy_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader

DEFAULT_CHECKPOINT = "/mnt/pqssd/pretrained/pi05_local_jax/params"


def checkpoint_root(path: str) -> pathlib.Path:
    resolved = pathlib.Path(path).expanduser().resolve()
    if resolved.name == "params" and (resolved / "_METADATA").is_file():
        resolved = resolved.parent
    if not (resolved / "params" / "_METADATA").is_file():
        raise FileNotFoundError(f"Expected an Orbax checkpoint at {resolved / 'params'}")
    return resolved


def scalar_int(value) -> int:
    return int(np.asarray(value).item())


def image_to_hwc_uint8(value) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 4:
        image = image[-1]
    if image.ndim != 3:
        raise ValueError(f"Expected RGB image, got {image.shape}")
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * (255 if image.max(initial=0) <= 1 else 1), 0, 255)
    return image.astype(np.uint8)


def augment_image(value, brightness: float, contrast: float, color: float, crop_ratio: float) -> np.ndarray:
    original = np.asarray(value)
    pil = Image.fromarray(image_to_hwc_uint8(original))
    if crop_ratio < 1:
        width, height = pil.size
        crop_width, crop_height = int(width * crop_ratio), int(height * crop_ratio)
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        pil = pil.crop((left, top, left + crop_width, top + crop_height)).resize((width, height), Image.BILINEAR)
    pil = ImageEnhance.Brightness(pil).enhance(brightness)
    pil = ImageEnhance.Contrast(pil).enhance(contrast)
    pil = ImageEnhance.Color(pil).enhance(color)
    result = np.asarray(pil, dtype=np.float32) / 255
    if original.shape[0] == 3:
        result = np.moveaxis(result, -1, 0)
    return result.astype(np.float32)


def make_observation(sample: dict) -> dict:
    observation = {
        key: np.asarray(value)
        for key, value in sample.items()
        if key.startswith("observation.images.") or key == "observation.state"
    }
    observation["prompt"] = str(sample["task"])
    return observation


@st.cache_resource(show_spinner="Loading 12 GB JAX Pi0.5 checkpoint...")
def load_policy(config_name: str, checkpoint: str, num_steps: int):
    train_config = _config.get_config(config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.norm_stats is None:
        raise FileNotFoundError(f"No norm stats found for {config_name}")
    started = time.perf_counter()
    policy = policy_config.create_trained_policy(
        train_config,
        checkpoint_root(checkpoint),
        norm_stats=data_config.norm_stats,
        sample_kwargs={"num_steps": num_steps},
    )
    return train_config, data_config, policy, time.perf_counter() - started


@st.cache_resource(show_spinner="Loading Kuavo LeRobot v3 episode...")
def load_dataset(config_name: str, dataset_root: str, repo_id: str, episode: int, video_backend: str):
    train_config = _config.get_config(config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    _data_loader.validate_lerobot_metadata(metadata, data_config)
    horizon = train_config.model.action_horizon
    dataset = LeRobotDataset(
        repo_id,
        root=dataset_root,
        episodes=[episode],
        delta_timestamps={"action": [step / metadata.fps for step in range(horizon)]},
        video_backend=video_backend,
    )
    return metadata, dataset


def inference_key(
    config_name: str,
    checkpoint: str,
    episode: int,
    frame: int,
    seed: int,
    num_steps: int,
    brightness: float,
    contrast: float,
    color: float,
    crop_ratio: float,
) -> tuple:
    return (config_name, checkpoint, episode, frame, seed, num_steps, brightness, contrast, color, crop_ratio)


def main() -> None:
    st.set_page_config(page_title="Kuavo OpenPI Pi0.5", layout="wide")
    st.title("Kuavo OpenPI Pi0.5 · Open-loop Viewer")
    st.caption("LeRobot v3 observations → JAX Pi0.5 → Kuavo absolute action chunk; no ROS or robot control.")

    with st.sidebar:
        config_name = st.selectbox("Kuavo config", ("pi05_kuavo", "pi05_kuavo_task2"))
        configured = _config.get_config(config_name)
        checkpoint = st.text_input("JAX checkpoint", DEFAULT_CHECKPOINT)
        dataset_root = st.text_input("Dataset root", str(configured.data.root))
        repo_id = st.text_input("Repo ID", str(configured.data.repo_id))
        episode = int(st.number_input("Episode", min_value=0, value=0, step=1))
        video_backend = st.selectbox("Video backend", ("torchcodec", "pyav", "video_reader"))
        num_steps = st.slider("Flow integration steps", 1, 20, 10)
        seed = int(st.number_input("Noise seed", min_value=0, value=0, step=1))
        require_cuda = st.checkbox("Require CUDA", value=True)
        st.divider()
        st.subheader("Visual perturbation")
        brightness = st.slider("Brightness", 0.4, 1.8, 1.0, 0.05)
        contrast = st.slider("Contrast", 0.4, 1.8, 1.0, 0.05)
        color = st.slider("Color", 0.0, 1.8, 1.0, 0.05)
        crop_ratio = st.slider("Center crop ratio", 0.5, 1.0, 1.0, 0.01)
        load_clicked = st.button("Load / run inference", type="primary", width="stretch")

    if load_clicked:
        st.session_state["kuavo_viewer_enabled"] = True
    if not st.session_state.get("kuavo_viewer_enabled", False):
        st.info("Configure the dataset and checkpoint, then click **Load / run inference**.")
        st.stop()

    devices = jax.devices()
    if require_cuda and not any(device.platform == "gpu" for device in devices):
        st.error(f"CUDA required, but JAX found: {devices}")
        st.stop()

    try:
        train_config, _, policy, load_seconds = load_policy(config_name, checkpoint, num_steps)
        metadata, dataset = load_dataset(config_name, dataset_root, repo_id, episode, video_backend)
    except Exception as exc:
        st.exception(exc)
        st.stop()

    max_frame = max(0, len(dataset) - 1)
    with st.sidebar:
        frame = st.slider("Frame in selected episode", 0, max_frame, 0)
        compare_horizon = st.slider("Compare horizon", 1, train_config.model.action_horizon, 50)

    sample = dict(dataset[frame])
    image_keys = sorted(key for key in sample if key.startswith("observation.images."))
    for key in image_keys:
        sample[key] = augment_image(sample[key], brightness, contrast, color, crop_ratio)

    key = inference_key(
        config_name,
        checkpoint,
        episode,
        frame,
        seed,
        num_steps,
        brightness,
        contrast,
        color,
        crop_ratio,
    )
    if st.session_state.get("kuavo_prediction_key") != key:
        rng = np.random.default_rng(seed + episode * 1_000_003 + frame)
        noise = rng.standard_normal(
            (train_config.model.action_horizon, train_config.model.action_dim), dtype=np.float32
        )
        started = time.perf_counter()
        result = policy.infer(make_observation(sample), noise=noise)
        st.session_state["kuavo_prediction"] = result
        st.session_state["kuavo_prediction_seconds"] = time.perf_counter() - started
        st.session_state["kuavo_prediction_key"] = key

    result = st.session_state["kuavo_prediction"]
    predicted = np.asarray(result["actions"], dtype=np.float32)
    target = np.asarray(sample["action"], dtype=np.float32)
    horizon = min(compare_horizon, len(predicted), len(target))
    valid = np.ones(horizon, dtype=bool)
    if "action_is_pad" in sample:
        valid &= ~np.asarray(sample["action_is_pad"], dtype=bool)[:horizon]
    predicted, target = predicted[:horizon][valid], target[:horizon][valid]
    error = predicted - target
    if not np.all(np.isfinite(predicted)):
        st.error("Prediction contains NaN or Inf")
        st.stop()

    progress = 0.0 if max_frame == 0 else frame / max_frame
    st.progress(progress, text=f"Episode {episode}, frame {frame}/{max_frame}")
    columns = st.columns(7)
    columns[0].metric("Device", str(devices[0]))
    columns[1].metric("Model load", f"{load_seconds:.2f} s")
    columns[2].metric("Inference", f"{st.session_state['kuavo_prediction_seconds'] * 1000:.1f} ms")
    columns[3].metric("Chunk", f"{len(result['actions'])} x {predicted.shape[-1]}")
    columns[4].metric("Valid horizon", len(predicted))
    columns[5].metric("First MAE", f"{np.abs(error[0]).mean():.5f}")
    columns[6].metric("Overall MAE", f"{np.abs(error).mean():.5f}")

    st.subheader("RGB observations")
    image_columns = st.columns(len(image_keys))
    for column, image_key in zip(image_columns, image_keys, strict=True):
        column.image(image_to_hwc_uint8(sample[image_key]), caption=image_key, width="stretch")

    st.caption(f"Prompt: {sample['task']}")
    action_names = metadata.features["action"]["names"]["action_names"]
    first_step = pd.DataFrame(
        {
            "dim": np.arange(predicted.shape[-1]),
            "name": action_names,
            "pred": predicted[0],
            "gt": target[0],
            "error": error[0],
            "abs_error": np.abs(error[0]),
        }
    )
    st.subheader("First action")
    st.dataframe(first_step, width="stretch", hide_index=True)

    chart_columns = st.columns(2)
    horizon_mae = np.abs(error).mean(axis=1)
    dim_mae = np.abs(error).mean(axis=0)
    with chart_columns[0]:
        st.subheader("Horizon MAE")
        st.line_chart(pd.DataFrame({"MAE": horizon_mae}))
    with chart_columns[1]:
        st.subheader("Per-dimension MAE")
        st.bar_chart(pd.DataFrame({"MAE": dim_mae}, index=action_names))

    dimension = st.selectbox(
        "Action dimension",
        range(predicted.shape[-1]),
        format_func=lambda index: f"{index}: {action_names[index]}",
    )
    trajectory = pd.DataFrame(
        {
            "pred": predicted[:, dimension],
            "gt": target[:, dimension],
            "error": error[:, dimension],
        }
    )
    st.line_chart(trajectory)

    action_stats = metadata.stats.get("action", {}) if metadata.stats else {}
    if "min" in action_stats and "max" in action_stats:
        lower, upper = np.asarray(action_stats["min"]), np.asarray(action_stats["max"])
        violation_rate = np.mean((predicted < lower) | (predicted > upper))
        st.metric("Dataset action-range violation", f"{violation_rate * 100:.2f}%")

    st.warning(
        "Open-loop on dataset observations is diagnostic only. The current local stats were generated for smoke testing; "
        "recompute full normalization stats before using MAE for model comparison."
    )


if __name__ == "__main__":
    main()

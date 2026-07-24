"""Read-only loader and transform smoke test for the supported SO-101 datasets."""

import argparse
import dataclasses
import pathlib

import lerobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.policies import so101_policy
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


def _describe(name: str, value) -> None:
    array = np.asarray(value)
    print(f"{name}: shape={array.shape}, dtype={array.dtype}, range=[{array.min():.6g}, {array.max():.6g}]")
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise AssertionError(f"{name} contains NaN or Inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", choices=("pi05_so101_60", "pi05_so101_90"), default="pi05_so101_60")
    parser.add_argument("--root", type=pathlib.Path)
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    data_factory = train_config.data
    root = args.root or pathlib.Path(data_factory.root or "")
    if not root.is_dir():
        raise FileNotFoundError(root)

    print(f"lerobot: version={lerobot.__version__}, module={pathlib.Path(lerobot.__file__).resolve()}")
    metadata = LeRobotDatasetMetadata(data_factory.repo_id, root=root)
    print(
        f"dataset: codebase={metadata.info['codebase_version']}, robot={metadata.info.get('robot_type')}, "
        f"fps={metadata.fps}, episodes={metadata.total_episodes}, frames={metadata.total_frames}"
    )
    print(f"cameras: {tuple(metadata.camera_keys)}")
    _data_loader.validate_lerobot_metadata(
        metadata,
        dataclasses.replace(
            data_factory.create_base_config(train_config.assets_dirs, train_config.model),
            expected_codebase_version="v3.0",
            expected_robot_type="so101_follower",
            expected_state_names=so101_policy.STATE_ACTION_NAMES,
            expected_action_names=so101_policy.STATE_ACTION_NAMES,
            required_camera_keys=so101_policy.CAMERA_KEYS,
        ),
    )

    horizon = train_config.model.action_horizon
    dataset = LeRobotDataset(
        data_factory.repo_id,
        root=root,
        delta_timestamps={"action": [step / metadata.fps for step in range(horizon)]},
    )
    frame = dataset[0]
    print(f"task: {frame['task']!r}")
    for key in so101_policy.CAMERA_KEYS:
        _describe(key, frame[key])
    _describe("observation.state", frame["observation.state"])
    _describe("action chunk", frame["action"])
    assert np.asarray(frame["action"]).shape == (horizon, 6)

    first_episode = metadata.episodes[0]
    episode_start = int(first_episode["dataset_from_index"])
    episode_end = int(first_episode["dataset_to_index"])
    boundary_index = episode_end - 1
    boundary_frame = dataset[boundary_index]
    query_indices, _ = dataset._get_query_indices(boundary_index, 0)  # noqa: SLF001
    assert all(episode_start <= index < episode_end for index in query_indices["action"])
    action_is_pad = np.asarray(boundary_frame["action_is_pad"])
    assert not action_is_pad[0]
    assert np.all(action_is_pad[1:])
    assert np.allclose(boundary_frame["action"], np.asarray(boundary_frame["action"])[-1])
    print(f"episode-end padding: {action_is_pad.sum()}/{horizon}; no cross-episode indices")

    transformed = _transforms.compose(
        [
            _transforms.PromptFromLeRobotTask(),
            *data_factory.data_transform_group().inputs,
            _transforms.ResizeImages(224, 224),
            _transforms.PadStatesAndActions(train_config.model.action_dim),
        ]
    )(frame)
    assert np.allclose(np.asarray(transformed["actions"])[..., :6], np.asarray(frame["action"]))
    transformed["tokenized_prompt"] = np.zeros((train_config.model.max_token_len,), dtype=np.int32)
    transformed["tokenized_prompt_mask"] = np.ones((train_config.model.max_token_len,), dtype=bool)
    model_batch = {
        key: (
            {nested_key: np.asarray(nested_value)[None, ...] for nested_key, nested_value in value.items()}
            if isinstance(value, dict)
            else np.asarray(value)[None, ...]
        )
        for key, value in transformed.items()
        if key != "prompt"
    }
    observation = _model.Observation.from_dict(model_batch)
    assert observation.state.shape == (1, 32)
    assert model_batch["actions"].shape == (1, horizon, 32)
    assert not bool(observation.image_masks["left_wrist_0_rgb"][0])
    assert bool(observation.image_masks["right_wrist_0_rgb"][0])
    for key, value in _transforms.flatten_dict(transformed).items():
        if key != "prompt":
            _describe(f"OpenPI {key}", value)
    print("PASS: SO-101 metadata, frame, chunk, boundary padding, absolute-action transform, and OpenPI shapes")


if __name__ == "__main__":
    main()

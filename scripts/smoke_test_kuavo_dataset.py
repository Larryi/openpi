"""Small, read-only smoke test for a local Kuavo LeRobot v3 dataset."""

import argparse
import pathlib

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training import config as _config


def _describe(name: str, value) -> None:
    array = np.asarray(value)
    print(f"{name}: shape={array.shape}, dtype={array.dtype}, range=[{array.min():.6g}, {array.max():.6g}]")


def _assert_finite(name: str, value) -> None:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.number) and not np.all(np.isfinite(array)):
        raise AssertionError(f"{name} contains NaN or Inf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("/mnt/pqssd/Real_PQ_3.0/TASK1_SZ_Repaired/lerobot_task1_345"),
    )
    parser.add_argument("--repo-id", default="kuavo_task1")
    parser.add_argument("--config-name", default="pi05_kuavo")
    args = parser.parse_args()

    metadata = LeRobotDatasetMetadata(args.repo_id, root=args.root)
    print(f"codebase_version: {metadata.info['codebase_version']}")
    print(f"fps: {metadata.fps}")
    print(f"episodes: {metadata.total_episodes}")
    print(f"frames: {metadata.total_frames}")
    print(f"features: {tuple(metadata.features)}")
    assert metadata.info["codebase_version"] == "v3.0"
    assert metadata.features["observation.state"]["shape"] == metadata.features["action"]["shape"]

    action_horizon = 50
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.root,
        delta_timestamps={"action": [step / metadata.fps for step in range(action_horizon)]},
    )
    frame = dataset[0]
    print(f"task: {frame['task']!r}")
    for key in metadata.camera_keys:
        _describe(key, frame[key])
    _describe("observation.state", frame["observation.state"])
    _describe("action chunk", frame["action"])
    assert np.asarray(frame["action"]).shape == (action_horizon, metadata.features["action"]["shape"][0])

    first_episode = metadata.episodes[0]
    episode_start = int(first_episode["dataset_from_index"])
    episode_end = int(first_episode["dataset_to_index"])
    boundary_index = episode_end - 1
    boundary_frame = dataset[boundary_index]
    query_indices, _ = dataset._get_query_indices(boundary_index, 0)  # noqa: SLF001
    assert all(episode_start <= index < episode_end for index in query_indices["action"])
    action_is_pad = np.asarray(boundary_frame["action_is_pad"])
    assert action_is_pad.shape == (action_horizon,)
    assert not action_is_pad[0]
    assert np.all(action_is_pad[1:])
    assert np.allclose(np.asarray(boundary_frame["action"]), np.asarray(boundary_frame["action"])[-1])
    print(f"episode-end padding: {action_is_pad.sum()}/{action_horizon}; no cross-episode indices")

    train_config = _config.get_config(args.config_name)
    data_factory = train_config.data
    if pathlib.Path(data_factory.root or "") != args.root:
        state_action_names = tuple(metadata.features["action"]["names"]["action_names"])
        delta_action_mask = tuple(not name.endswith("_claw") for name in state_action_names)
        data_factory = _config.LeRobotKuavoDataConfig(
            repo_id=args.repo_id,
            root=str(args.root),
            action_dim=metadata.features["action"]["shape"][0],
            state_action_names=state_action_names,
            camera_keys=tuple(metadata.camera_keys),
            delta_action_mask=delta_action_mask,
        )
    data_transforms = data_factory.data_transform_group()

    transformed = _transforms.compose(
        [
            _transforms.PromptFromLeRobotTask(),
            *data_transforms.inputs,
            _transforms.ResizeImages(224, 224),
            _transforms.PadStatesAndActions(train_config.model.action_dim),
        ]
    )(dataset[0])
    raw_state = np.asarray(dataset[0]["observation.state"])
    raw_actions = np.asarray(dataset[0]["action"])
    transformed_actions = np.asarray(transformed["actions"])[..., : data_factory.action_dim]
    mask = np.asarray(data_factory.delta_action_mask)
    assert np.allclose(transformed_actions[..., mask], raw_actions[..., mask] - raw_state[mask])
    assert np.allclose(transformed_actions[..., ~mask], raw_actions[..., ~mask])
    print(f"delta mask: {tuple(mask)} (grippers remain absolute)")
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

    print(f"prompt: {transformed['prompt']!r}")
    for key, image in observation.images.items():
        print(f"OpenPI image[{key}]: shape={image.shape}, mask={bool(observation.image_masks[key])}")
        assert image.shape == (1, 224, 224, 3)
    print(f"OpenPI state: shape={observation.state.shape}")
    print(f"OpenPI actions: shape={model_batch['actions'].shape}")
    assert observation.state.shape == (1, train_config.model.action_dim)
    assert model_batch["actions"].shape == (1, action_horizon, train_config.model.action_dim)
    assert bool(observation.image_masks["left_wrist_0_rgb"][0]) == (
        "observation.images.wrist_cam_l" in metadata.camera_keys
    )
    assert bool(observation.image_masks["right_wrist_0_rgb"][0]) == (
        "observation.images.wrist_cam_r" in metadata.camera_keys
    )

    for key, value in _transforms.flatten_dict(transformed).items():
        if key != "prompt":
            _assert_finite(key, value)
    print("PASS: Gates A-C")


if __name__ == "__main__":
    main()

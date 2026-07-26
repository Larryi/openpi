"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import dataclasses
import pathlib

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


class StateActionOnlyInputs(transforms.DataTransformFn):
    """Extract normalization inputs without touching image or prompt features."""

    def __call__(self, x: dict) -> dict:
        return {
            "state": np.asarray(x["observation.state"], dtype=np.float32),
            "actions": np.asarray(x["action"], dtype=np.float32),
        }


def disable_video_decoding(dataset) -> None:
    """Disable LeRobot video reads recursively for state/action-only statistics."""
    while isinstance(dataset, _data_loader.TransformedDataset):
        dataset = dataset.dataset
    if isinstance(dataset, _data_loader.WeightedLeRobotDataset):
        for child in dataset.datasets:
            disable_video_decoding(child)
        return
    if hasattr(dataset, "_query_videos"):
        dataset._query_videos = lambda query_timestamps, ep_idx: {}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
    state_action_only: bool = False,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    if state_action_only:
        disable_video_decoding(dataset)
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    num_samples = num_batches * batch_size
    sampler = _data_loader.create_weighted_sampler(dataset, num_samples=num_samples)
    transforms_to_apply = (
        [
            StateActionOnlyInputs(),
            *data_config.data_transforms.inputs[1:],
        ]
        if state_action_only
        else [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ]
    )
    dataset = _data_loader.TransformedDataset(dataset, transforms_to_apply)
    shuffle = sampler is None and max_frames is not None and max_frames < len(dataset)
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        sampler=sampler,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(
    config_name: str,
    max_frames: int | None = None,
    dataset_root: str | None = None,
    tokenizer_path: str | None = None,
    assets_base_dir: str | None = None,
    batch_size: int | None = None,
    num_workers: int | None = None,
    state_action_only: bool = False,
):
    config = _config.get_config(config_name)
    if assets_base_dir is not None:
        config = dataclasses.replace(config, assets_base_dir=assets_base_dir)
    data_overrides = {}
    if dataset_root is not None:
        data_overrides["root"] = str(pathlib.Path(dataset_root).expanduser().resolve())
    if tokenizer_path is not None:
        data_overrides["tokenizer_path"] = str(pathlib.Path(tokenizer_path).expanduser().resolve())
    if data_overrides:
        config = dataclasses.replace(config, data=dataclasses.replace(config.data, **data_overrides))
    if batch_size is not None or num_workers is not None:
        config = dataclasses.replace(
            config,
            batch_size=batch_size or config.batch_size,
            num_workers=num_workers if num_workers is not None else config.num_workers,
        )
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
            max_frames,
            state_action_only,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)

"""Smoke-test a weighted Kuavo LeRobot mixture without starting model training."""

import argparse
import collections
import math

import numpy as np

import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        default="pi05_kuavo_task1_mixed",
        choices=("pi05_kuavo_task1_mixed", "pi05_kuavo_task2_mixed"),
    )
    parser.add_argument("--samples", type=int, default=2_000)
    parser.add_argument("--decode-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    dataset = _data_loader.create_torch_dataset(
        data_config,
        train_config.model.action_horizon,
        train_config.model,
    )
    if not isinstance(dataset, _data_loader.TransformedDataset):
        raise AssertionError("Expected the prompt transform wrapper")
    mixture = dataset.dataset
    if not isinstance(mixture, _data_loader.WeightedLeRobotDataset):
        raise AssertionError("Expected a weighted LeRobot mixture")

    print(f"config={args.config_name} total_frames={len(mixture)} action_horizon={train_config.model.action_horizon}")
    for source, source_dataset in zip(mixture.sources, mixture.datasets, strict=True):
        metadata = source_dataset.meta
        loaded_episodes = {int(index) for index in source_dataset.hf_dataset["episode_index"]}
        if source.episodes is not None and loaded_episodes != set(source.episodes):
            raise AssertionError(f"Episode subset mismatch for {source.name}")
        print(
            f"source={source.name} weight={source.weight:.3f} frames={len(source_dataset)} "
            f"episodes={len(metadata.episodes) if source.episodes is None else len(source.episodes)} "
            f"fps={metadata.fps} version={metadata.info.get('codebase_version')} root={source.root}"
        )
        episode_indices = source_dataset.hf_dataset["episode_index"]
        boundary = next(
            (index for index in range(len(episode_indices) - 1) if episode_indices[index] != episode_indices[index + 1]),
            len(episode_indices) - 1,
        )
        boundary_item = source_dataset[boundary]
        action_chunk = np.asarray(boundary_item["action"])
        action_is_pad = np.asarray(boundary_item["action_is_pad"])
        if not action_is_pad[1:].all():
            raise AssertionError(f"Action chunk crosses an episode boundary for {source.name}")
        if not np.allclose(action_chunk[1:], action_chunk[:1]):
            raise AssertionError(f"Episode-tail actions are not padded with the final action for {source.name}")
        print(
            f"boundary={source.name} episode={int(boundary_item['episode_index'])} "
            f"chunk={action_chunk.shape} padded={int(action_is_pad.sum())}/{len(action_is_pad)}"
        )

    sampler = _data_loader.create_weighted_sampler(mixture, num_samples=args.samples, seed=args.seed)
    counts = collections.Counter(mixture.source_index(index) for index in sampler)
    for source_index, source in enumerate(mixture.sources):
        actual = counts[source_index] / args.samples
        tolerance = max(0.02, 4 * math.sqrt(source.weight * (1 - source.weight) / args.samples))
        print(f"sampled={source.name} expected={source.weight:.3f} actual={actual:.3f}")
        if abs(actual - source.weight) > tolerance:
            raise AssertionError(f"Sampling ratio for {source.name} is outside tolerance {tolerance:.3f}")

    decode_sampler = _data_loader.create_weighted_sampler(
        mixture,
        num_samples=args.decode_samples,
        seed=args.seed + 1,
    )
    for index in decode_sampler:
        item = dataset[index]
        state = np.asarray(item["observation.state"])
        actions = np.asarray(item["action"])
        if actions.shape[0] != train_config.model.action_horizon:
            raise AssertionError(f"Unexpected action chunk shape: {actions.shape}")
        for key, value in item.items():
            array = np.asarray(value)
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                raise AssertionError(f"NaN/Inf in {key}")
        print(
            f"decoded_source={mixture.sources[mixture.source_index(index)].name} "
            f"state={state.shape}/{state.dtype} actions={actions.shape}/{actions.dtype} "
            f"prompt={item.get('prompt')!r}"
        )

    transformed = _data_loader.transform_dataset(dataset, data_config, skip_norm_stats=True)
    torch_loader = _data_loader.TorchDataLoader(
        transformed,
        local_batch_size=1,
        sampler=_data_loader.create_weighted_sampler(mixture, num_samples=1, seed=args.seed + 2),
        num_batches=1,
    )
    observation, actions = next(
        iter(
            _data_loader.DataLoaderImpl(
                data_config,
                torch_loader,
            )
        )
    )
    if actions.shape != (1, train_config.model.action_horizon, train_config.model.action_dim):
        raise AssertionError(f"Unexpected transformed action shape: {actions.shape}")
    if not all(np.isfinite(np.asarray(value)).all() for value in (*observation.images.values(), observation.state, actions)):
        raise AssertionError("NaN/Inf after OpenPI transforms")
    print(f"transformed_state={observation.state.shape} transformed_actions={actions.shape} PASS")


if __name__ == "__main__":
    main()

import dataclasses
import json
import types

import jax
import numpy as np

from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import data_loader as _data_loader


class _IndexDataset:
    def __init__(self, size: int, offset: int):
        self._size = size
        self._offset = offset

    def __getitem__(self, index):
        return self._offset + index

    def __len__(self):
        return self._size


def test_weighted_lerobot_dataset_maps_indices_and_source_probabilities():
    sources = (
        _config.LeRobotDatasetConfig(name="main", repo_id="main", weight=0.55),
        _config.LeRobotDatasetConfig(name="slave", repo_id="slave", weight=0.20),
        _config.LeRobotDatasetConfig(name="old", repo_id="old", weight=0.25),
    )
    dataset = _data_loader.WeightedLeRobotDataset(
        (_IndexDataset(10, 0), _IndexDataset(20, 100), _IndexDataset(70, 200)),
        sources,
    )

    assert len(dataset) == 100
    assert dataset[9] == 9
    assert dataset[10] == 100
    assert dataset[29] == 119
    assert dataset[30] == 200
    assert dataset.source_index(9) == 0
    assert dataset.source_index(10) == 1
    assert dataset.source_index(30) == 2

    sampler = _data_loader.create_weighted_sampler(dataset, num_samples=20_000, seed=7)
    counts = np.bincount([dataset.source_index(index) for index in sampler], minlength=3)
    np.testing.assert_allclose(counts / counts.sum(), [0.55, 0.20, 0.25], atol=0.015)

    wrapped = _data_loader.TransformedDataset(
        _data_loader.TransformedDataset(dataset, []),
        [],
    )
    assert _data_loader.create_weighted_sampler(wrapped, num_samples=10, seed=7) is not None


def test_kuavo_mixed_configs_preserve_episode_domains():
    task1 = _config.get_config("pi05_kuavo_task1_mixed").data.lerobot_datasets
    task2 = _config.get_config("pi05_kuavo_task2_mixed").data.lerobot_datasets

    assert [source.name for source in task1] == ["beijing_main", "beijing_slave", "suzhou_repaired"]
    assert [source.weight for source in task1] == [0.55, 0.20, 0.25]
    assert task1[0].episodes == tuple(range(104))
    assert task1[1].episodes == tuple(range(104, 220))
    assert task2[0].episodes == tuple(range(102))
    assert task2[1].episodes == tuple(range(102, 208))


def test_kuavo_runtime_mix_parses_resolved_hf_sources(monkeypatch):
    payload = [
        {"name": "sz", "repo_id": "owner/sz", "root": "/datasets/sz", "weight": 0.25},
        {"name": "bj", "repo_id": "owner/bj", "root": "/datasets/bj", "weight": 0.75},
    ]
    monkeypatch.setenv("KUAVO_DATASET_MIX_JSON", json.dumps(payload))
    monkeypatch.setenv("KUAVO_MIX_ASSET_ID", "kuavo_task1_mix_123")

    sources = _config._kuavo_mix_from_env()
    assert [(source.repo_id, source.root, source.weight) for source in sources] == [
        ("owner/sz", "/datasets/sz", 0.25),
        ("owner/bj", "/datasets/bj", 0.75),
    ]
    assert _config._kuavo_mix_primary_root("/fallback") == "/datasets/sz"
    assert _config._kuavo_mix_asset_id("fallback") == "kuavo_task1_mix_123"


def test_validate_lerobot_metadata_accepts_list_feature_names():
    names = ["joint_1.pos", "gripper.pos"]
    metadata = types.SimpleNamespace(
        info={"codebase_version": "v3.0", "robot_type": "test_robot"},
        features={
            "observation.state": {"names": names},
            "action": {"names": names},
        },
        camera_keys=["observation.images.front"],
    )
    data_config = _config.DataConfig(
        root="/dataset",
        expected_codebase_version="v3.0",
        expected_robot_type="test_robot",
        expected_state_names=names,
        expected_action_names=names,
        required_camera_keys=("observation.images.front",),
    )

    _data_loader.validate_lerobot_metadata(metadata, data_config)


def test_torch_data_loader():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 16)

    loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=4,
        num_batches=2,
    )
    batches = list(loader)

    assert len(batches) == 2
    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_torch_data_loader_infinite():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 4)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4)
    data_iter = iter(loader)

    for _ in range(10):
        _ = next(data_iter)


def test_torch_data_loader_parallel():
    config = pi0_config.Pi0Config(action_dim=24, action_horizon=50, max_token_len=48)
    dataset = _data_loader.FakeDataset(config, 10)

    loader = _data_loader.TorchDataLoader(dataset, local_batch_size=4, num_batches=2, num_workers=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == 4 for x in jax.tree.leaves(batch))


def test_with_fake_dataset():
    config = _config.get_config("debug")

    loader = _data_loader.create_data_loader(config, skip_norm_stats=True, num_batches=2)
    batches = list(loader)

    assert len(batches) == 2

    for batch in batches:
        assert all(x.shape[0] == config.batch_size for x in jax.tree.leaves(batch))

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)


def test_with_real_dataset():
    config = _config.get_config("pi0_aloha_sim")
    config = dataclasses.replace(config, batch_size=4)

    loader = _data_loader.create_data_loader(
        config,
        # Skip since we may not have the data available.
        skip_norm_stats=True,
        num_batches=2,
        shuffle=True,
    )
    # Make sure that we can get the data config.
    assert loader.data_config().repo_id == config.data.repo_id

    batches = list(loader)

    assert len(batches) == 2

    for _, actions in batches:
        assert actions.shape == (config.batch_size, config.model.action_horizon, config.model.action_dim)

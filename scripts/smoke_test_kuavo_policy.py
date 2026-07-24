"""Load a Kuavo checkpoint through serve_policy and run one offline observation."""

import argparse
import pathlib

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import serve_policy

from openpi.training import config as _config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="pi05_kuavo_smoke")
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        default=pathlib.Path("checkpoints/pi05_kuavo_smoke/repaired_task1/9"),
    )
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=pathlib.Path("/mnt/pqssd/Real_PQ_3.0/TASK1_SZ_Repaired/lerobot_task1_345"),
    )
    parser.add_argument("--repo-id", default="kuavo_task1")
    args = parser.parse_args()

    train_config = _config.get_config(args.config_name)
    policy = serve_policy.create_policy(
        serve_policy.Args(
            policy=serve_policy.Checkpoint(config=args.config_name, dir=str(args.checkpoint.resolve())),
        )
    )

    frame = LeRobotDataset(args.repo_id, root=args.root)[0]
    observation = {
        key: value
        for key, value in frame.items()
        if key.startswith("observation.images.") or key == "observation.state"
    }
    observation["prompt"] = frame["task"]
    result = policy.infer(observation)

    actions = np.asarray(result["actions"])
    robot_action_dim = train_config.data.action_dim
    print(f"model action shape: ({train_config.model.action_horizon}, {train_config.model.action_dim})")
    print(f"server action shape: {actions.shape}")
    print(f"server action range: [{actions.min():.6g}, {actions.max():.6g}]")
    assert actions.shape == (train_config.model.action_horizon, robot_action_dim)
    assert np.all(np.isfinite(actions))
    print("PASS: Gate G")


if __name__ == "__main__":
    main()

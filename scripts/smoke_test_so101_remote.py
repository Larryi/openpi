#!/usr/bin/env python3
"""Send one SO101 dataset observation through an OpenPI WebSocket server."""

from __future__ import annotations

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy


def dataset_image(value) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Expected an image, got {image.shape}")
    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    return image_tools.convert_to_uint8(image)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--dataset-root", type=Path, default=Path("/mnt/pqssd/so101/datasets/merged_grab_blue_pen_60")
    )
    parser.add_argument("--repo-id", default="so101_grab_blue_pen_60")
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    frame = LeRobotDataset(args.repo_id, root=args.dataset_root.expanduser().resolve())[args.frame]
    request = {
        "observation.images.front": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(dataset_image(frame["observation.images.front"]), 224, 224)
        ),
        "observation.images.wrist": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(dataset_image(frame["observation.images.wrist"]), 224, 224)
        ),
        "observation.state": np.asarray(frame["observation.state"], dtype=np.float32),
        "prompt": str(frame["task"]),
    }
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    result = client.infer(request)
    actions = np.asarray(result["actions"], dtype=np.float32)
    print("server metadata:", client.get_server_metadata())
    print("actions shape:", actions.shape)
    print("actions min/max:", actions.min(axis=0), actions.max(axis=0))
    print("policy timing:", result.get("policy_timing"))
    assert actions.shape == (50, 6)
    assert np.isfinite(actions).all()
    print("PASS: lerobot_hil -> OpenPI WebSocket inference")


if __name__ == "__main__":
    main()

import dataclasses

import einops
import numpy as np

from openpi import transforms

STATE_ACTION_NAMES = (
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
)
CAMERA_KEYS = (
    "observation.images.front",
    "observation.images.wrist",
)


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size == 0 or float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError(f"Expected an unbatched RGB image, got shape {image.shape}")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got shape {image.shape}")
    return image


@dataclasses.dataclass(frozen=True)
class SO101Inputs(transforms.DataTransformFn):
    """Map a two-camera SO-101 LeRobot observation into OpenPI model inputs."""

    def __call__(self, data: dict) -> dict:
        front = _parse_image(data["observation.images.front"])
        wrist = _parse_image(data["observation.images.wrist"])
        inputs = {
            "image": {
                "base_0_rgb": front,
                "left_wrist_0_rgb": np.zeros_like(front),
                "right_wrist_0_rgb": wrist,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": np.asarray(data["observation.state"], dtype=np.float32),
        }
        if "action" in data:
            inputs["actions"] = np.asarray(data["action"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]
        return inputs


@dataclasses.dataclass(frozen=True)
class SO101Outputs(transforms.DataTransformFn):
    """Trim OpenPI padding and return absolute SO-101 position targets."""

    action_dim: int = len(STATE_ACTION_NAMES)

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}

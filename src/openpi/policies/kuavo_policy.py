import dataclasses

import einops
import numpy as np

from openpi import transforms

TASK1_STATE_ACTION_NAMES = (
    "zarm_r1_link",
    "zarm_r2_link",
    "zarm_r3_link",
    "zarm_r4_link",
    "zarm_r5_link",
    "zarm_r6_link",
    "zarm_r7_link",
    "right_claw",
)
TASK2_STATE_ACTION_NAMES = (
    "zarm_l1_link",
    "zarm_l2_link",
    "zarm_l3_link",
    "zarm_l4_link",
    "zarm_l5_link",
    "zarm_l6_link",
    "zarm_l7_link",
    "left_claw",
    *TASK1_STATE_ACTION_NAMES,
)
TASK1_CAMERA_KEYS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_r",
)
TASK2_CAMERA_KEYS = (
    "observation.images.head_cam_h",
    "observation.images.wrist_cam_l",
    "observation.images.wrist_cam_r",
)
TASK1_DELTA_ACTION_MASK = (True,) * 7 + (False,)
TASK2_DELTA_ACTION_MASK = (True,) * 7 + (False,) + (True,) * 7 + (False,)


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
class KuavoInputs(transforms.DataTransformFn):
    """Map Kuavo LeRobot/environment observations into OpenPI model inputs."""

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation.images.head_cam_h"])

        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}
        for model_key, dataset_key in (
            ("left_wrist_0_rgb", "observation.images.wrist_cam_l"),
            ("right_wrist_0_rgb", "observation.images.wrist_cam_r"),
        ):
            if dataset_key in data:
                images[model_key] = _parse_image(data[dataset_key])
                image_masks[model_key] = np.True_
            else:
                images[model_key] = np.zeros_like(base_image)
                image_masks[model_key] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
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
class KuavoOutputs(transforms.DataTransformFn):
    """Trim model action padding and return Kuavo joint-position commands."""

    action_dim: int = 8

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[..., : self.action_dim]}

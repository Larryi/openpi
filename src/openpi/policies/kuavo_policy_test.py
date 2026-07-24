import numpy as np

from openpi.policies import kuavo_policy


def test_kuavo_inputs_maps_available_cameras_and_pads_missing_camera():
    data = {
        "observation.images.head_cam_h": np.ones((3, 20, 30), dtype=np.float32),
        "observation.images.wrist_cam_r": np.zeros((3, 20, 30), dtype=np.float32),
        "observation.state": np.arange(8, dtype=np.float32),
        "action": np.ones((50, 8), dtype=np.float32),
        "prompt": "pick up the object",
    }

    result = kuavo_policy.KuavoInputs()(data)

    assert result["image"]["base_0_rgb"].shape == (20, 30, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert result["image"]["base_0_rgb"].max() == 255
    assert result["image"]["left_wrist_0_rgb"].shape == (20, 30, 3)
    assert not result["image_mask"]["left_wrist_0_rgb"]
    assert result["image_mask"]["right_wrist_0_rgb"]
    assert result["state"].shape == (8,)
    assert result["actions"].shape == (50, 8)
    assert result["prompt"] == "pick up the object"


def test_kuavo_outputs_removes_model_padding():
    result = kuavo_policy.KuavoOutputs(action_dim=8)({"actions": np.ones((50, 32))})
    assert result["actions"].shape == (50, 8)


def test_task2_layout_and_delta_mask_keep_both_grippers_absolute():
    assert len(kuavo_policy.TASK2_STATE_ACTION_NAMES) == 16
    assert kuavo_policy.TASK2_STATE_ACTION_NAMES[7] == "left_claw"
    assert kuavo_policy.TASK2_STATE_ACTION_NAMES[15] == "right_claw"
    assert not kuavo_policy.TASK2_DELTA_ACTION_MASK[7]
    assert not kuavo_policy.TASK2_DELTA_ACTION_MASK[15]
    assert sum(kuavo_policy.TASK2_DELTA_ACTION_MASK) == 14

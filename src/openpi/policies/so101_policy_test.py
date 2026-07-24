import numpy as np

from openpi.policies import so101_policy


def test_so101_inputs_maps_two_cameras_and_keeps_absolute_actions():
    actions = np.arange(300, dtype=np.float32).reshape(50, 6)
    data = {
        "observation.images.front": np.ones((3, 20, 30), dtype=np.float32),
        "observation.images.wrist": np.zeros((3, 20, 30), dtype=np.float32),
        "observation.state": np.arange(6, dtype=np.float32),
        "action": actions,
        "task": "Grab the blue pen",
    }

    result = so101_policy.SO101Inputs()(data)

    assert result["image"]["base_0_rgb"].shape == (20, 30, 3)
    assert result["image"]["base_0_rgb"].dtype == np.uint8
    assert result["image"]["base_0_rgb"].max() == 255
    assert result["image"]["right_wrist_0_rgb"].shape == (20, 30, 3)
    assert result["image_mask"]["base_0_rgb"]
    assert not result["image_mask"]["left_wrist_0_rgb"]
    assert result["image_mask"]["right_wrist_0_rgb"]
    assert np.array_equal(result["actions"], actions)
    assert result["prompt"] == "Grab the blue pen"


def test_so101_outputs_removes_model_padding():
    result = so101_policy.SO101Outputs()({"actions": np.ones((50, 32), dtype=np.float32)})
    assert result["actions"].shape == (50, 6)

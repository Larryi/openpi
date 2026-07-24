from types import SimpleNamespace
from typing import ClassVar

import numpy as np

from scripts import run_so101_openpi_client as client_module


class _FakeRobot:
    def __init__(self):
        self.sent = []

    def get_observation(self):
        observation = dict.fromkeys(client_module.ACTION_NAMES, 0.0)
        observation["front"] = np.zeros((8, 8, 3), dtype=np.uint8)
        observation["wrist"] = np.zeros((8, 8, 3), dtype=np.uint8)
        return observation

    def send_action(self, action):
        self.sent.append(action)


class _FakeAsyncWorker:
    instances: ClassVar[list] = []

    def __init__(self, host, port):
        del host, port
        self.submissions = 0
        self.requests = []
        self._result = None
        self.__class__.instances.append(self)

    @property
    def busy(self):
        return False

    def start(self):
        pass

    def submit(self, request):
        assert request["observation.state"].shape == (6,)
        self.submissions += 1
        self.requests.append(request)
        actions = np.full((50, 6), 10.0 * self.submissions, dtype=np.float32)
        self._result = client_module.AsyncInferenceResult({"actions": actions}, latency_s=0.001)
        return True

    def poll(self):
        result, self._result = self._result, None
        return result

    def close(self):
        pass


def test_async_continuous_prefetches_and_switches_chunks(monkeypatch):
    _FakeAsyncWorker.instances.clear()
    monkeypatch.setattr(client_module, "AsyncPolicyWorker", _FakeAsyncWorker)
    robot = _FakeRobot()
    args = SimpleNamespace(
        host="localhost",
        port=8000,
        control_fps=100_000.0,
        open_loop_horizon=5,
        steps=12,
        async_prefetch_threshold=3,
        async_transition_steps=0,
        async_result_timeout=1.0,
        prompt="test",
        no_clip_training_range=True,
    )

    client_module.run_async_continuous(robot, args, np.zeros((50, 6), dtype=np.float32))

    sent = np.asarray([[action[name] for name in client_module.ACTION_NAMES] for action in robot.sent])
    assert sent.shape == (12, 6)
    np.testing.assert_array_equal(sent[:5], 0.0)
    np.testing.assert_array_equal(sent[5:10], 10.0)
    np.testing.assert_array_equal(sent[10:], 20.0)
    assert _FakeAsyncWorker.instances[0].submissions == 2


def test_blend_chunk_smooths_only_requested_prefix():
    actions = np.full((5, 2), 10.0, dtype=np.float32)
    blended = client_module.blend_chunk(actions, np.zeros(2, dtype=np.float32), transition_steps=3)

    np.testing.assert_allclose(blended[:, 0], [2.5, 5.0, 7.5, 10.0, 10.0])


def test_rtc_continuous_sends_guided_requests_and_replaces_queue(monkeypatch):
    _FakeAsyncWorker.instances.clear()
    monkeypatch.setattr(client_module, "AsyncPolicyWorker", _FakeAsyncWorker)
    robot = _FakeRobot()
    args = SimpleNamespace(
        host="localhost",
        port=8000,
        control_fps=100_000.0,
        open_loop_horizon=5,
        steps=8,
        rtc_queue_threshold=4,
        rtc_execution_horizon=3,
        rtc_max_guidance_weight=10.0,
        async_result_timeout=1.0,
        prompt="test",
        no_clip_training_range=True,
    )

    client_module.run_rtc_continuous(robot, args, np.zeros((50, 6), dtype=np.float32))

    worker = _FakeAsyncWorker.instances[0]
    assert len(robot.sent) == 8
    assert worker.submissions > 1
    assert worker.requests[0]["rtc"]["prev_actions"].shape == (3, 6)
    assert worker.requests[0]["rtc"]["execution_horizon"] == 3
    assert worker.requests[0]["rtc"]["max_guidance_weight"] == 10.0

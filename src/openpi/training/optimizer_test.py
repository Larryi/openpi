import numpy as np
import pytest

from openpi.training import optimizer


def test_cosine_decay_schedule_default_is_unchanged():
    schedule = optimizer.CosineDecaySchedule().create()

    np.testing.assert_allclose(schedule(30_000), 2.5e-6, rtol=1e-5)
    np.testing.assert_allclose(schedule(50_000), 2.5e-6, rtol=1e-5)


def test_cosine_decay_schedule_continuous_tail():
    schedule = optimizer.CosineDecaySchedule(
        tail_start_step=44_000,
        tail_decay_steps=6_000,
        tail_decay_lr=2.5e-7,
    ).create()

    np.testing.assert_allclose(schedule(43_999), 2.5e-6, rtol=1e-5)
    np.testing.assert_allclose(schedule(44_000), 2.5e-6, rtol=1e-5)
    np.testing.assert_allclose(schedule(47_000), 1.375e-6, rtol=1e-5)
    np.testing.assert_allclose(schedule(50_000), 2.5e-7, rtol=1e-5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tail_start_step": 44_000},
        {"tail_start_step": 20_000, "tail_decay_steps": 6_000, "tail_decay_lr": 2.5e-7},
        {"tail_start_step": 44_000, "tail_decay_steps": 0, "tail_decay_lr": 2.5e-7},
        {"tail_start_step": 44_000, "tail_decay_steps": 6_000, "tail_decay_lr": 3e-6},
    ],
)
def test_cosine_decay_schedule_rejects_invalid_tail(kwargs):
    with pytest.raises(ValueError):
        optimizer.CosineDecaySchedule(**kwargs)

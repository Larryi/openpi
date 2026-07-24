import dataclasses
from typing import Protocol, runtime_checkable

import jax.numpy as jnp
import optax

import openpi.shared.array_typing as at


@runtime_checkable
class LRScheduleConfig(Protocol):
    def create(self) -> optax.Schedule: ...


@dataclasses.dataclass(frozen=True)
class CosineDecaySchedule(LRScheduleConfig):
    """Cosine decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 2.5e-5
    decay_steps: int = 30_000
    decay_lr: float = 2.5e-6
    # Optional continuation tail. This keeps the original schedule unchanged through
    # tail_start_step, then starts a second cosine at decay_lr and reaches tail_decay_lr
    # after tail_decay_steps. It is intended for continuous checkpoint continuation.
    tail_start_step: int | None = None
    tail_decay_steps: int | None = None
    tail_decay_lr: float | None = None

    def __post_init__(self) -> None:
        tail_values = (self.tail_start_step, self.tail_decay_steps, self.tail_decay_lr)
        if any(value is not None for value in tail_values) and not all(value is not None for value in tail_values):
            raise ValueError("tail_start_step, tail_decay_steps, and tail_decay_lr must be set together")
        if self.tail_start_step is not None:
            if self.tail_start_step < self.decay_steps:
                raise ValueError("tail_start_step must be greater than or equal to decay_steps")
            if self.tail_decay_steps is None or self.tail_decay_steps <= 0:
                raise ValueError("tail_decay_steps must be positive")
            if self.tail_decay_lr is None or not 0 <= self.tail_decay_lr <= self.decay_lr:
                raise ValueError("tail_decay_lr must be between 0 and decay_lr")

    def create(self) -> optax.Schedule:
        base_schedule = optax.warmup_cosine_decay_schedule(
            init_value=self.peak_lr / (self.warmup_steps + 1),
            peak_value=self.peak_lr,
            warmup_steps=self.warmup_steps,
            decay_steps=self.decay_steps,
            end_value=self.decay_lr,
        )
        if self.tail_start_step is None:
            return base_schedule

        tail_schedule = optax.cosine_decay_schedule(
            init_value=self.decay_lr,
            decay_steps=self.tail_decay_steps,
            alpha=self.tail_decay_lr / self.decay_lr,
        )
        return optax.join_schedules([base_schedule, tail_schedule], [self.tail_start_step])


@dataclasses.dataclass(frozen=True)
class RsqrtDecaySchedule(LRScheduleConfig):
    """Inverse square root decay schedule with warmup."""

    warmup_steps: int = 1_000
    peak_lr: float = 5e-5
    timescale: float = 10_000

    def create(self) -> optax.Schedule:
        return optax.join_schedules(
            [
                optax.linear_schedule(
                    init_value=self.peak_lr / (self.warmup_steps + 1),
                    end_value=self.peak_lr,
                    transition_steps=self.warmup_steps,
                ),
                lambda step: self.peak_lr / jnp.sqrt((self.timescale + step) / self.timescale),
            ],
            [self.warmup_steps],
        )


@runtime_checkable
class OptimizerConfig(Protocol):
    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation: ...


@dataclasses.dataclass(frozen=True)
class AdamW(OptimizerConfig):
    """AdamW optimizer."""

    b1: float = 0.9
    b2: float = 0.95
    eps: float = 1e-8
    # Changing this to 0 can cause out-of-memory errors for some reason, so we set it to a negligible value.
    weight_decay: float = 1e-10
    clip_gradient_norm: float = 1.0

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        tx = optax.adamw(
            lr, b1=self.b1, b2=self.b2, eps=self.eps, weight_decay=self.weight_decay, mask=weight_decay_mask
        )

        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


@dataclasses.dataclass(frozen=True)
class SGD(OptimizerConfig):
    """SGD optimizer."""

    lr: float = 5e-5
    momentum: float = 0.9
    nesterov: bool = False

    def create(
        self,
        lr: optax.ScalarOrSchedule,
        weight_decay_mask: at.PyTree | None = None,
    ) -> optax.GradientTransformation:
        assert weight_decay_mask is None, "Weight decay is not supported for SGD"
        return optax.sgd(lr, momentum=self.momentum, nesterov=self.nesterov)


def create_optimizer(
    optimizer: OptimizerConfig, lr_schedule: LRScheduleConfig, weight_decay_mask: at.PyTree | None = None
) -> optax.GradientTransformation:
    lr = lr_schedule.create()
    return optimizer.create(lr, weight_decay_mask=weight_decay_mask)

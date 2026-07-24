import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import openpi.models.pi0 as _pi0
import openpi.models.pi0_config as _pi0_config
from openpi.shared import nnx_utils


def _get_frozen_state(config: _pi0_config.Pi0Config) -> nnx.State:
    abstract_model = nnx.eval_shape(config.create, jax.random.key(0))

    freeze_filter = config.get_freeze_filter()
    return nnx.state(abstract_model, nnx.All(nnx.Param, freeze_filter)).flat_state()


def test_pi0_full_finetune():
    config = _pi0_config.Pi0Config()
    state = _get_frozen_state(config)
    assert len(state) == 0


def test_balanced_remat_policy():
    common = {
        "paligemma_variant": "dummy",
        "action_expert_variant": "dummy",
        "vision_variant": "mu/14",
    }
    baseline = nnx.state(nnx.eval_shape(_pi0_config.Pi0Config(**common).create, jax.random.key(0))).to_pure_dict()
    balanced = nnx.state(
        nnx.eval_shape(
            _pi0_config.Pi0Config(**common, remat_policy="dots_with_no_batch_dims_saveable").create,
            jax.random.key(0),
        )
    ).to_pure_dict()
    assert jax.tree.map(lambda value: value.shape, baseline) == jax.tree.map(lambda value: value.shape, balanced)


def test_invalid_remat_policy():
    with pytest.raises(ValueError, match="Unknown JAX checkpoint policy"):
        _pi0_config.Pi0Config(remat_policy="not_a_policy")


def test_rtc_prefix_weights():
    weights = jax.jit(_pi0.get_rtc_prefix_weights, static_argnums=2)(
        np.asarray(2, dtype=np.int32), np.asarray(6, dtype=np.int32), 10
    )
    np.testing.assert_allclose(weights, [1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.0, 0.0, 0.0])


def test_rtc_sample_actions_dummy_model():
    config = _pi0_config.Pi0Config(
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        vision_variant="mu/14",
        action_dim=6,
        action_horizon=8,
        max_token_len=16,
        pi05=True,
    )
    key = jax.random.key(0)
    model = config.create(key)
    observation = config.fake_obs(batch_size=1)
    noise = jnp.ones((1, config.action_horizon, config.action_dim), dtype=jnp.float32)
    actions = nnx_utils.module_jit(model.sample_actions)(
        key,
        observation,
        num_steps=2,
        noise=noise,
        rtc_prev_actions=jnp.zeros_like(noise),
        rtc_inference_delay=jnp.asarray(1, dtype=jnp.int32),
        rtc_execution_horizon=jnp.asarray(4, dtype=jnp.int32),
        rtc_max_guidance_weight=jnp.asarray(10.0, dtype=jnp.float32),
    )

    assert actions.shape == noise.shape
    assert np.isfinite(actions).all()


def test_pi0_gemma_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora")
    state = _get_frozen_state(config)
    assert len(state) == 9
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    assert all("_1" not in p for p in state)


def test_pi0_action_expert_lora():
    config = _pi0_config.Pi0Config(action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # excluding embedder, rest of the params should be same as gemma_lora.
    assert len(state) == 8
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)
    # all frozen params should have _1 in their path since it's the action expert.
    assert all(any("_1" in p for p in path) for path in state)


def test_pi0_all_lora():
    config = _pi0_config.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    state = _get_frozen_state(config)
    # sum of gemma_lora and action_expert_lora's frozen params.
    assert len(state) == 17
    assert all("lora" not in p for p in state)
    assert all("llm" in p for p in state)

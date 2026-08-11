import numpy as np
import pytest

from scripts.experiment3.dlolab_wrapping.replay_winding_audit import (
    semantic_winding_transition_audit,
    signed_winding_turns_np,
    validate_official_run_config,
    winding_loss_from_signed_turns,
)


def _required():
    return {
        "n_envs": 100,
        "max_iter": 51,
        "n_steps": 10,
        "n_steps_sub": 10,
        "act_dim": 12,
        "popsize": 400,
        "seed": 123,
        "bound": 0.1,
        "l2_bound": 0.1,
        "angle_bound": 1.0,
        "angle_scale": [1.0, 1.0, 1.0],
        "sigma": 0.05,
        "use_last_state_reward": False,
        "randomized_args": None,
    }


def _run_config():
    return {
        "n_envs": 100,
        "n_steps": 10,
        "n_steps_sub": 10,
        "act_dim": 12,
        "popsize": 400,
        "sigma0": 0.05,
        "per_comp_bound": 0.1,
        "l2_bound": 0.1,
        "angle_bound": 1.0,
        "angle_scale": [1.0, 1.0, 1.0],
        "max_iters": 51,
        "seed": 123,
        "use_last_state_reward": False,
        "randomized_args": None,
    }


def test_full_effective_official_config_passes():
    validate_official_run_config(_run_config(), _required())


@pytest.mark.parametrize(
    "field,bad",
    [
        ("act_dim", 6),
        ("popsize", 200),
        ("sigma0", 0.1),
        ("per_comp_bound", 0.2),
        ("l2_bound", 0.2),
        ("angle_bound", 2.0),
        ("angle_scale", [1.0, 0.5, 1.0]),
        ("use_last_state_reward", True),
        ("randomized_args", {"pos_bound": [-0.1, -0.1, 0.1, 0.1]}),
    ],
)
def test_effective_official_config_mismatch_is_rejected(field, bad):
    run = _run_config()
    run[field] = bad
    with pytest.raises(RuntimeError):
        validate_official_run_config(run, _required())


def _circle(reverse=False):
    theta = np.linspace(0.0, 2.0 * np.pi, 100, endpoint=False)
    if reverse:
        theta = theta[::-1]
    return np.stack(
        [np.cos(theta), np.sin(theta), np.zeros_like(theta)],
        axis=1,
    )


def test_signed_winding_is_orientation_sensitive():
    post = np.array([[0.0, 0.0, 0.0]])
    a = signed_winding_turns_np(_circle(False), post)[0]
    b = signed_winding_turns_np(_circle(True), post)[0]
    assert np.isclose(abs(a), 1.0, atol=1e-10)
    assert np.isclose(abs(b), 1.0, atol=1e-10)
    assert np.sign(a) == -np.sign(b)


def test_published_winding_loss_zero_at_one_turn():
    assert np.isclose(
        winding_loss_from_signed_turns(np.array([1.0, -1.0, 1.0])),
        0.0,
        atol=1e-12,
    )


def test_semantic_transition_detects_unwrapped_to_wrapped():
    magnitude = np.array([
        [0.05, 0.02, 0.01],
        [0.10, 0.03, 0.02],
        [0.40, 0.05, 0.02],
        [0.82, 0.07, 0.03],
    ])
    result = semantic_winding_transition_audit(
        magnitude,
        unwrapped_max=0.25,
        wrapped_min=0.75,
        require_later=True,
    )
    assert result["transition_post_count"] == 1


def test_micro_variation_does_not_count():
    magnitude = np.array([
        [0.000001, 0.0, 0.0],
        [0.000003, 0.0, 0.0],
        [0.000006, 0.0, 0.0],
    ])
    result = semantic_winding_transition_audit(
        magnitude,
        unwrapped_max=0.25,
        wrapped_min=0.75,
        require_later=True,
    )
    assert result["transition_post_count"] == 0


def test_reverse_transition_does_not_count_when_order_required():
    magnitude = np.array([
        [0.90, 0.0, 0.0],
        [0.80, 0.0, 0.0],
        [0.20, 0.0, 0.0],
    ])
    result = semantic_winding_transition_audit(
        magnitude,
        unwrapped_max=0.25,
        wrapped_min=0.75,
        require_later=True,
    )
    assert result["transition_post_count"] == 0



@pytest.mark.parametrize(
    "missing_field",
    [
        "use_last_state_reward",
        "randomized_args",
    ],
)
def test_missing_semantic_official_config_field_is_rejected(missing_field):
    run = _run_config()
    del run[missing_field]

    with pytest.raises(RuntimeError):
        validate_official_run_config(
            run,
            _required(),
        )

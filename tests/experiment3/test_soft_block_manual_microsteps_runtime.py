from scripts.experiment3.phase0c_hidden_dynamics.common import load_config
from state_diff.env.block_pushing.soft_block_pushing import SoftBlockPushEnv

CONFIG = "configs/experiment3/hlf_sbp_phase0c_dynamics.json"


def test_one_outer_step_is_exactly_eight_true_microsteps(monkeypatch):
    config = load_config(CONFIG)
    config["execution"]["pre_snapshot_settle_outer_steps"] = 1
    env = SoftBlockPushEnv(config)
    try:
        internal_calls = step_calls = 0
        original_internal = env.soft_block.apply_internal_forces
        original_step = env.pybullet_client.stepSimulation
        def internal():
            nonlocal internal_calls
            internal_calls += 1
            return original_internal()
        def step():
            nonlocal step_calls
            step_calls += 1
            return original_step()
        monkeypatch.setattr(env.soft_block, "apply_internal_forces", internal)
        monkeypatch.setattr(env.pybullet_client, "stepSimulation", step)
        before = env.physics_step
        env.step_one_physics()
        assert internal_calls == step_calls == 8
        assert env.physics_step == before + 1
        assert env.pybullet_client.getPhysicsEngineParameters()["numSubSteps"] == 1
    finally:
        env.close()

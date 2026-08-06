import numpy as np

from state_diff.env.block_pushing.kelvin_voigt_soft_block import SpringStepStats
from state_diff.env.block_pushing.manual_microstep_integrator import (
    ManualMicrostepConfig, ManualMicrostepIntegrator)


class FakeClient:
    def __init__(self): self.events = []
    def setTimeStep(self, value): self.dt = value
    def setPhysicsEngineParameter(self, **kwargs): self.parameters = kwargs
    def stepSimulation(self): self.events.append("step")


class FakeBlock:
    def __init__(self, client): self.client, self.calls = client, 0
    def apply_internal_forces(self):
        self.calls += 1; self.client.events.append("internal")
        return SpringStepStats(
            {"structural": float(self.calls), "shear": 0., "bending": 0.},
            self.calls + .5, self.calls, self.calls % 2, 512, 1e-14,
            .01, .02)


def test_manual_microstep_order_time_and_aggregation():
    client = FakeClient(); block = FakeBlock(client)
    config = ManualMicrostepConfig(1 / 240, 8)
    integrator = ManualMicrostepIntegrator(client, config)
    callback_dts = []
    def external(index, count, dt):
        client.events.append("external"); callback_dts.append((index, count, dt))
    stats = integrator.advance_outer_step(block, external)
    assert np.isclose(config.micro_timestep_s * 8, config.outer_timestep_s)
    assert np.isclose(client.dt, (1 / 240) / 8)
    assert client.parameters["numSubSteps"] == 1
    assert block.calls == len(callback_dts) == 8
    assert client.events == [item for _ in range(8)
                             for item in ("internal", "external", "step")]
    assert stats.final_energy_by_kind_j["structural"] == 8
    assert stats.capped_force_count == 4
    assert stats.force_evaluation_count == 512 * 8
    assert stats.max_uncapped_edge_force_n == 8.5


def test_manual_microstep_config_rejects_nonpositive_values():
    for config in (ManualMicrostepConfig(0, 1), ManualMicrostepConfig(1, 0)):
        try:
            config.validate()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid manual microstep config accepted")

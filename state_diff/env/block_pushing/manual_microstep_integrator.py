"""True force-recomputed manual microstep integration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional


@dataclass(frozen=True)
class ManualMicrostepConfig:
    """Outer interval and number of equal manual mechanics substeps."""

    outer_timestep_s: float
    microsteps_per_outer: int

    def validate(self) -> None:
        """Reject non-positive time or count."""
        if self.outer_timestep_s <= 0:
            raise ValueError("outer_timestep_s must be positive")
        if self.microsteps_per_outer <= 0:
            raise ValueError("microsteps_per_outer must be positive")

    @property
    def micro_timestep_s(self) -> float:
        """Return the Bullet timestep for one manual microstep."""
        return self.outer_timestep_s / self.microsteps_per_outer


@dataclass(frozen=True)
class OuterStepMechanicsStats:
    """Aggregated force telemetry for one outer 240 Hz interval."""

    final_energy_by_kind_j: Dict[str, float]
    max_uncapped_edge_force_n: float
    max_applied_edge_force_n: float
    capped_force_count: int
    force_evaluation_count: int
    max_net_internal_force_residual_n: float
    min_edge_length_m: float
    max_edge_length_m: float
    microsteps_per_outer: int


ExternalLoadCallback = Callable[[int, int, float], None]


def configure_client_for_manual_microsteps(
    client, config: ManualMicrostepConfig,
) -> None:
    """Configure one Bullet client for a manual microstep loop."""
    config.validate()
    client.setTimeStep(config.micro_timestep_s)
    client.setPhysicsEngineParameter(numSubSteps=1)


class ManualMicrostepIntegrator:
    """Recompute internal and external forces before every microstep."""

    def __init__(self, client, config: ManualMicrostepConfig):
        config.validate()
        self.client = client
        self.config = config
        configure_client_for_manual_microsteps(client, config)

    def advance_outer_step(
        self, soft_block,
        external_load_callback: Optional[ExternalLoadCallback] = None,
    ) -> OuterStepMechanicsStats:
        """Advance exactly one outer interval and aggregate microstep stats."""
        samples = []
        for micro_index in range(self.config.microsteps_per_outer):
            stats = soft_block.apply_internal_forces()
            if external_load_callback is not None:
                external_load_callback(
                    micro_index, self.config.microsteps_per_outer,
                    self.config.micro_timestep_s)
            self.client.stepSimulation()
            samples.append(stats)
        return OuterStepMechanicsStats(
            final_energy_by_kind_j=dict(samples[-1].energy_by_kind_j),
            max_uncapped_edge_force_n=max(
                sample.max_uncapped_edge_force_n for sample in samples),
            max_applied_edge_force_n=max(
                sample.max_applied_edge_force_n for sample in samples),
            capped_force_count=sum(sample.capped_force_count for sample in samples),
            force_evaluation_count=sum(
                sample.force_evaluation_count for sample in samples),
            max_net_internal_force_residual_n=max(
                sample.net_internal_force_residual_n for sample in samples),
            min_edge_length_m=min(sample.min_edge_length_m for sample in samples),
            max_edge_length_m=max(sample.max_edge_length_m for sample in samples),
            microsteps_per_outer=self.config.microsteps_per_outer)

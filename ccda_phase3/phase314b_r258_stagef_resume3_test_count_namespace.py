"""Frozen test-count namespace for Phase3.14b-r2.5.8 Stage F Resume3.

This module deliberately separates historical Stage-D Resume2 counts from the
current Stage-F Resume2 porcelain-test count.  It contains no scientific model,
feature, fold, integrator, threshold, or data-selection logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple


EXPECTED_REGULAR_PASSED = 1056
EXPECTED_HISTORICAL_CLOSED_WORLD_PASSED = 8
EXPECTED_STAGE_D_PASSED = 84
EXPECTED_STAGE_D_RESUME1_PASSED = 40
EXPECTED_STAGED_RESUME2_PASSED = 55
EXPECTED_STAGE_E_PASSED = 81
EXPECTED_STAGE_F_PASSED = 84
EXPECTED_STAGE_F_RESUME1_PASSED = 47
EXPECTED_STAGEF_RESUME2_PASSED = 48

EXPECTED_TEMPORAL_FILES = 58
EXPECTED_TEMPORAL_PASSED = 1503

HISTORICAL_STAGE_D_RESUME2_NAME = "EXPECTED_STAGED_RESUME2_PASSED"
CURRENT_STAGE_F_RESUME2_NAME = "EXPECTED_STAGEF_RESUME2_PASSED"


@dataclass(frozen=True)
class TemporalPopulation:
    regular: int = EXPECTED_REGULAR_PASSED
    historical_closed_world: int = EXPECTED_HISTORICAL_CLOSED_WORLD_PASSED
    stage_d: int = EXPECTED_STAGE_D_PASSED
    stage_d_resume1: int = EXPECTED_STAGE_D_RESUME1_PASSED
    stage_d_resume2: int = EXPECTED_STAGED_RESUME2_PASSED
    stage_e: int = EXPECTED_STAGE_E_PASSED
    stage_f: int = EXPECTED_STAGE_F_PASSED
    stage_f_resume1: int = EXPECTED_STAGE_F_RESUME1_PASSED
    stage_f_resume2: int = EXPECTED_STAGEF_RESUME2_PASSED

    def ordered_items(self) -> Tuple[Tuple[str, int], ...]:
        return (
            ("regular", self.regular),
            ("historical_closed_world", self.historical_closed_world),
            ("stage_d", self.stage_d),
            ("stage_d_resume1", self.stage_d_resume1),
            ("stage_d_resume2", self.stage_d_resume2),
            ("stage_e", self.stage_e),
            ("stage_f", self.stage_f),
            ("stage_f_resume1", self.stage_f_resume1),
            ("stage_f_resume2", self.stage_f_resume2),
        )

    def as_dict(self) -> Dict[str, int]:
        return dict(self.ordered_items())

    def total_passed(self) -> int:
        return sum(value for _, value in self.ordered_items())

    def validate(self) -> None:
        if HISTORICAL_STAGE_D_RESUME2_NAME == CURRENT_STAGE_F_RESUME2_NAME:
            raise RuntimeError("historical/current Resume2 constant names collide")
        if self.stage_d_resume2 != 55:
            raise RuntimeError(
                f"historical Stage-D Resume2 count changed: {self.stage_d_resume2}"
            )
        if self.stage_f_resume2 != 48:
            raise RuntimeError(
                f"current Stage-F Resume2 count changed: {self.stage_f_resume2}"
            )
        if self.total_passed() != EXPECTED_TEMPORAL_PASSED:
            raise RuntimeError(
                "temporal passed-count arithmetic changed: "
                f"{self.total_passed()} != {EXPECTED_TEMPORAL_PASSED}"
            )


def validate_observed_counts(observed: Mapping[str, int]) -> None:
    """Validate an observed temporal-population mapping exactly and in order."""

    contract = TemporalPopulation()
    contract.validate()
    expected = contract.ordered_items()
    actual = tuple((str(key), int(value)) for key, value in observed.items())
    if actual != expected:
        raise RuntimeError(
            "temporal population changed:\n"
            f"expected={expected!r}\n"
            f"actual={actual!r}"
        )


TemporalPopulation().validate()

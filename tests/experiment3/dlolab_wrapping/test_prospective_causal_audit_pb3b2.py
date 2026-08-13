import inspect

import pytest

from scripts.experiment3.dlolab_wrapping.snapshot_causal_audit_pb3 import PB3Blocked
from scripts.experiment3.dlolab_wrapping.prospective_causal_audit_pb3b2 import (
    LIVE_FAILED,
    SNAPSHOT_FAILED,
    enforce_live_cohort_barrier,
    enforce_snapshot_restore_barrier,
    execute_with_evidence,
)


def _live(valid=True):
    return [
        {"pair_id": f"pair_{i}", "valid": bool(valid or i != 9)}
        for i in range(10)
    ]


def _restores(valid=True):
    return [
        {
            "rollout_id": rollout,
            "time_index": 13 if rollout % 2 == 0 else 20,
            "repeat_index": repeat,
            "valid": bool(valid or (rollout, repeat) != (19, 2)),
        }
        for rollout in range(20)
        for repeat in range(3)
    ]


def test_live_barrier_requires_exactly_10_without_failure():
    enforce_live_cohort_barrier(_live())
    with pytest.raises(PB3Blocked) as caught:
        enforce_live_cohort_barrier(_live(False))
    assert caught.value.verdict == LIVE_FAILED
    assert caught.value.details["causal_future_status"] == "UNTESTED"
    assert caught.value.details["snapshot_capture_count"] == 0


def test_snapshot_barrier_requires_20_branches_times_3_restores():
    enforce_snapshot_restore_barrier(_restores())
    with pytest.raises(PB3Blocked) as caught:
        enforce_snapshot_restore_barrier(_restores(False))
    assert caught.value.verdict == SNAPSHOT_FAILED
    assert caught.value.details["causal_future_status"] == "UNTESTED"
    assert caught.value.details["future_suffix_executed"] is False


def test_snapshot_barrier_rejects_missing_or_duplicate_branch():
    rows = _restores()
    rows[-3:] = rows[:3]
    with pytest.raises(PB3Blocked):
        enforce_snapshot_restore_barrier(rows)


def test_execution_order_places_both_global_barriers_before_future():
    source = inspect.getsource(execute_with_evidence)
    live_barrier = source.index("enforce_live_cohort_barrier(live_records)")
    snapshot_barrier = source.index("_validate_all_snapshot_restores(")
    future = source.index("_run_future_suffixes(")
    assert live_barrier < snapshot_barrier < future

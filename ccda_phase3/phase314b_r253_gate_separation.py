"""Gate-separation audit for Phase3.14b-r2.5.3.

This additive train-only module separates three contracts that were previously
collapsed into one Boolean:

* exact four-horizon, 87-dimensional reconstruction fidelity;
* cable branch-identity transport;
* full-reverse train-only candidate quality.

It does not change training, rows, seeds, objectives, schedulers, thresholds,
or the DeformableRavens submodule.  It retrains the three immutable r2.5.2
diagnostic models and attributes the reconstruction failure by state group,
future horizon, source, timestep, and noise ID.  Synthetic controls verify that
separating the gates does not admit branch swaps, branch collapse, or topology
corruption.
"""

from __future__ import annotations

import hashlib
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ccda_phase3.phase314b_r21_geometry import calibrated_validity
from ccda_phase3.phase314b_r22_geometry import (
    nearest_index_metrics,
    torch_inverse_standardize,
)
from ccda_phase3.phase314b_r231_controls import (
    RANDOM_SINGLE_BRANCH_GATE,
    ReconstructionGate,
    gate_reconstruction,
    target_reconstruction_metrics,
)
from ccda_phase3.phase314b_r241_multirow import LabeledTupleBank
from ccda_phase3.phase314b_r242_frozen_prior import paired_branch_audit
import ccda_phase3.phase314b_r252_transport_attribution as r252
from ccda_phase3.schema_v2 import DEFAULT_TF, N_BEADS, ROBOT_PROXY_DIM, STATE_DIM

PHASE = "phase3_14b_r253"
BASE_REPORT_COMMIT = "674ff08e49882de6de6f0ed977b39ef53b98fea1"
BASE_IMPLEMENTATION_COMMIT = "3ec513ec4bbde6c8017dfbb4694cc9824129b063"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
EXPECTED_CONTRACT_SHA256 = (
    "fa2725ca40da2499008360f13d291d2ce8694e6393910b0522fe800f4f37cdcc"
)
EXPECTED_R252_ROOT_CAUSE = (
    "phase314b_r252_composite_one_step_gate_conflation_supported"
)
EXPECTED_R252_MECHANISMS = (
    "composite_gate_conflation",
    "per_timestep_gradient_miscalibration",
)
EXPECTED_R252_RECOMMENDATION = None
EXPECTED_PAIRED_ROWS = r252.EXPECTED_PAIRED_ROWS
DIAGNOSTIC_OBJECTIVE_NAMES = r252.DIAGNOSTIC_OBJECTIVE_NAMES
PAIR_TIMESTEPS = r252.PAIR_TIMESTEPS
EVALUATION_NOISE_SEEDS = r252.EVALUATION_NOISE_SEEDS
COMMON_PAIRED_PRIOR_SEED = r252.COMMON_PAIRED_PRIOR_SEED
COMMON_PAIRED_TRAINING_SEED = r252.COMMON_PAIRED_TRAINING_SEED

CABLE_DIM = N_BEADS * 2
if CABLE_DIM != 48 or CABLE_DIM + ROBOT_PROXY_DIM != STATE_DIM:
    raise RuntimeError("unexpected state-v2 dimension contract")

GATE_SEPARATION_SCHEMA = "phase314b_r253_gate_separation_v1"
RECONSTRUCTION_DECOMPOSITION_SCHEMA = (
    "phase314b_r253_reconstruction_decomposition_v1"
)
SYNTHETIC_CONTROL_SCHEMA = "phase314b_r253_synthetic_gate_controls_v1"
HISTORICAL_REPRODUCTION_SCHEMA = "phase314b_r253_r252_one_step_reproduction_v1"

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r253_gate_separation.py",
    "scripts/phase3_14b_r253_preflight.py",
    "scripts/phase3_14b_r253_run_pilot.py",
    "scripts/phase3_14b_r253_finalize.py",
    "scripts/phase3_14b_r253_run.sh",
    "tests/test_phase314b_r253_gate_separation.py",
)

DEPENDENCY_PATHS = (
    "ccda_phase3/phase314b_r231_controls.py",
    "ccda_phase3/phase314b_r242_frozen_prior.py",
    "ccda_phase3/phase314b_r251_gradient_calibration.py",
    "ccda_phase3/phase314b_r252_transport_attribution.py",
    "scripts/phase3_14b_r252_run_pilot.py",
    "scripts/phase3_14b_r252_finalize.py",
    "reports/phase3_14b_r252_pilot_summary.json",
    "reports/phase3_14b_r252_summary.json",
    "reports/phase3_14b_r252_report.md",
)


@dataclass(frozen=True)
class GateSeparationSpec:
    """Fixed audit-only thresholds.

    Reconstruction thresholds are inherited verbatim from
    ``RANDOM_SINGLE_BRANCH_GATE``.  Branch thresholds are inherited verbatim
    from ``paired_branch_audit``.  The physical/candidate thresholds are used
    only to characterize the already-committed train-only reverse evidence;
    they do not select a configuration.
    """

    branch_own_fraction_min: float = 0.90
    branch_separation_p50_min: float = 0.50
    branch_cosine_p50_min: float = 0.50
    one_step_physical_validity_min: float = 0.875
    reverse_sample_validity_min: float = 0.875
    reverse_valid_query_min: float = 0.875
    reverse_both_branch_support_min: float = 0.75
    topology_inversion_p95_max: float = 0.21739130434782608
    reproduction_absolute_tolerance: float = 2.0e-6
    robot_proxy_offset_z: float = 1.0
    cable_translation_m: float = 0.015

    def validate(self) -> None:
        for name, value in asdict(self).items():
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError(f"invalid gate-separation value {name}")
        for name in (
            "branch_own_fraction_min",
            "one_step_physical_validity_min",
            "reverse_sample_validity_min",
            "reverse_valid_query_min",
            "reverse_both_branch_support_min",
            "topology_inversion_p95_max",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0,1]")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(root: Path) -> Dict[str, str]:
    base = Path(root)
    return {path: sha256_file(base / path) for path in SOURCE_PATHS}


def dependency_sha256(root: Path) -> Dict[str, str]:
    base = Path(root)
    return {path: sha256_file(base / path) for path in DEPENDENCY_PATHS}


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=Path(root), text=True).strip()


def assert_only_allowed_worktree_paths(
    root: Path,
    allowed_paths: Sequence[str],
) -> None:
    allowed = {str(Path(path).as_posix()) for path in allowed_paths}
    output = git_output(root, "status", "--porcelain", "--untracked-files=all")
    observed: set[str] = set()
    for line in output.splitlines():
        if not line:
            continue
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        observed.add(str(Path(value).as_posix()))
    unexpected = sorted(observed - allowed)
    if unexpected:
        raise RuntimeError("unexpected worktree changes: " + ", ".join(unexpected))


def _as_numpy(value: torch.Tensor | np.ndarray, *, name: str) -> np.ndarray:
    array = (
        value.detach().cpu().numpy()
        if torch.is_tensor(value)
        else np.asarray(value)
    )
    array = np.asarray(array, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _stats(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise ValueError("statistics require a finite non-empty vector")
    return {
        "min": float(np.min(array)),
        "p05": float(np.percentile(array, 5)),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def reconstruction_gate_components(
    metrics: Mapping[str, Any],
    gate: ReconstructionGate = RANDOM_SINGLE_BRANCH_GATE,
) -> Dict[str, Any]:
    gate.validate()
    values = {
        "z_mse": float(metrics["z_mse"]),
        "ordered_rmse_p95": float(metrics["ordered_rmse"]["p95"]),
        "segment_relative_error_p95": float(
            metrics["segment_relative_error"]["p95"]
        ),
        "chain_relative_error_p95": float(
            metrics["chain_relative_error"]["p95"]
        ),
    }
    thresholds = {
        "z_mse": float(gate.z_mse_max),
        "ordered_rmse_p95": float(gate.ordered_rmse_p95_max),
        "segment_relative_error_p95": float(
            gate.segment_relative_error_p95_max
        ),
        "chain_relative_error_p95": float(
            gate.chain_relative_error_p95_max
        ),
    }
    passes = {name: bool(values[name] <= thresholds[name]) for name in values}
    return {
        "values": values,
        "thresholds": thresholds,
        "passes": passes,
        "failed_components": [name for name, passed in passes.items() if not passed],
        "z_state_gate_pass": passes["z_mse"],
        "cable_geometry_gate_pass": bool(
            passes["ordered_rmse_p95"]
            and passes["segment_relative_error_p95"]
            and passes["chain_relative_error_p95"]
        ),
        "historical_exact_reconstruction_pass": bool(all(passes.values())),
    }


def _group_z_metrics(
    predicted_z: np.ndarray,
    target_z: np.ndarray,
    active_mask: np.ndarray,
) -> Dict[str, Any]:
    if predicted_z.shape != target_z.shape:
        raise ValueError("z prediction/target shape mismatch")
    if predicted_z.ndim != 3 or predicted_z.shape[1:] != (DEFAULT_TF, STATE_DIM):
        raise ValueError("standardized futures must be [N,4,87]")
    active = np.asarray(active_mask, dtype=bool)
    if active.shape != (DEFAULT_TF, STATE_DIM):
        raise ValueError("active mask must be [4,87]")
    error2 = np.square(predicted_z - target_z)
    full_mask = np.broadcast_to(active[None], error2.shape)
    cable_active = active.copy()
    cable_active[:, CABLE_DIM:] = False
    robot_active = active.copy()
    robot_active[:, :CABLE_DIM] = False
    cable_mask = np.broadcast_to(cable_active[None], error2.shape)
    robot_mask = np.broadcast_to(robot_active[None], error2.shape)

    def mse(mask: np.ndarray) -> float:
        count = int(np.sum(mask))
        if count == 0:
            return 0.0
        return float(np.sum(error2[mask]) / count)

    full_count = int(np.sum(full_mask))
    cable_sse = float(np.sum(error2[cable_mask]))
    robot_sse = float(np.sum(error2[robot_mask]))
    total_sse = cable_sse + robot_sse
    return {
        "full_z_mse": mse(full_mask),
        "cable_z_mse": mse(cable_mask),
        "robot_proxy_z_mse": mse(robot_mask),
        "full_active_count": full_count,
        "cable_active_count": int(np.sum(cable_mask)),
        "robot_proxy_active_count": int(np.sum(robot_mask)),
        "cable_weighted_contribution": float(cable_sse / max(full_count, 1)),
        "robot_proxy_weighted_contribution": float(robot_sse / max(full_count, 1)),
        "cable_error_fraction": float(cable_sse / max(total_sse, 1.0e-20)),
        "robot_proxy_error_fraction": float(robot_sse / max(total_sse, 1.0e-20)),
    }


def _single_horizon_metrics(
    *,
    predicted_z: np.ndarray,
    target_z: np.ndarray,
    predicted_raw: np.ndarray,
    target_raw: np.ndarray,
    active_mask: np.ndarray,
    horizon: int,
    gate: ReconstructionGate,
) -> Dict[str, Any]:
    if not 0 <= int(horizon) < DEFAULT_TF:
        raise ValueError("horizon out of range")
    h = int(horizon)
    pz = predicted_z[:, h]
    tz = target_z[:, h]
    pr = predicted_raw[:, h]
    tr = target_raw[:, h]
    active = np.asarray(active_mask[h], dtype=bool)
    error2 = np.square(pz - tz)
    full_z_mse = float(np.mean(error2[:, active]))
    cable_active = active[:CABLE_DIM]
    robot_active = active[CABLE_DIM:]
    cable_z_mse = (
        float(np.mean(error2[:, :CABLE_DIM][:, cable_active]))
        if bool(np.any(cable_active))
        else 0.0
    )
    robot_z_mse = (
        float(np.mean(error2[:, CABLE_DIM:][:, robot_active]))
        if bool(np.any(robot_active))
        else 0.0
    )
    pxy = pr[:, :CABLE_DIM].reshape(-1, N_BEADS, 2)
    txy = tr[:, :CABLE_DIM].reshape(-1, N_BEADS, 2)
    ordered = np.sqrt(np.mean(np.square(pxy - txy), axis=(1, 2)))
    pedge = pxy[:, 1:] - pxy[:, :-1]
    tedge = txy[:, 1:] - txy[:, :-1]
    psegment = np.linalg.norm(pedge, axis=-1)
    tsegment = np.linalg.norm(tedge, axis=-1)
    segment_relative = np.abs(psegment - tsegment) / np.maximum(tsegment, 1.0e-6)
    pchain = np.sum(psegment, axis=-1)
    tchain = np.sum(tsegment, axis=-1)
    chain_relative = np.abs(pchain - tchain) / np.maximum(tchain, 1.0e-6)
    values = {
        "z_mse": full_z_mse,
        "cable_z_mse": cable_z_mse,
        "robot_proxy_z_mse": robot_z_mse,
        "ordered_rmse": _stats(ordered.tolist()),
        "segment_relative_error": _stats(segment_relative.reshape(-1).tolist()),
        "chain_relative_error": _stats(chain_relative.reshape(-1).tolist()),
    }
    passes = {
        "z_mse": bool(full_z_mse <= gate.z_mse_max),
        "ordered_rmse_p95": bool(
            values["ordered_rmse"]["p95"] <= gate.ordered_rmse_p95_max
        ),
        "segment_relative_error_p95": bool(
            values["segment_relative_error"]["p95"]
            <= gate.segment_relative_error_p95_max
        ),
        "chain_relative_error_p95": bool(
            values["chain_relative_error"]["p95"]
            <= gate.chain_relative_error_p95_max
        ),
    }
    return {
        "horizon": h,
        "row_count": int(predicted_z.shape[0]),
        "metrics": values,
        "passes": passes,
        "full_gate_pass": bool(all(passes.values())),
        "cable_geometry_gate_pass": bool(
            passes["ordered_rmse_p95"]
            and passes["segment_relative_error_p95"]
            and passes["chain_relative_error_p95"]
        ),
    }


def _selected_reconstruction(
    *,
    predicted_z: np.ndarray,
    target_z: np.ndarray,
    predicted_raw: np.ndarray,
    target_raw: np.ndarray,
    active_mask: np.ndarray,
    index: np.ndarray,
    gate: ReconstructionGate,
) -> Dict[str, Any]:
    selected = np.asarray(index, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        raise ValueError("reconstruction subset cannot be empty")
    metrics = target_reconstruction_metrics(
        predicted_z=predicted_z[selected],
        target_z=target_z[selected],
        predicted_raw=predicted_raw[selected],
        target_raw=target_raw[selected],
        active_mask=active_mask,
    )
    return {
        "row_count": int(selected.size),
        "metrics": metrics,
        "components": reconstruction_gate_components(metrics, gate),
        "gate_pass": bool(gate_reconstruction(metrics, gate)),
    }


def _grouped_reconstruction(
    *,
    predicted_z: np.ndarray,
    target_z: np.ndarray,
    predicted_raw: np.ndarray,
    target_raw: np.ndarray,
    active_mask: np.ndarray,
    labels: np.ndarray,
    gate: ReconstructionGate,
) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    passes: List[bool] = []
    for label in sorted(set(labels.tolist())):
        index = np.flatnonzero(labels == label)
        item = _selected_reconstruction(
            predicted_z=predicted_z,
            target_z=target_z,
            predicted_raw=predicted_raw,
            target_raw=target_raw,
            active_mask=active_mask,
            index=index,
            gate=gate,
        )
        values[str(label)] = item
        passes.append(bool(item["gate_pass"]))
    return {
        "groups": values,
        "group_count": int(len(values)),
        "all_group_gate_pass": bool(all(passes)),
        "group_pass_fraction": float(np.mean(np.asarray(passes, dtype=np.float64))),
    }


def _pair_relative_error(
    *,
    predicted_raw: np.ndarray,
    target_raw: np.ndarray,
    bank: LabeledTupleBank,
    source_pair_ids: Sequence[int],
) -> Dict[str, Any]:
    pair_ids = np.asarray(source_pair_ids, dtype=np.int64)
    source = bank.source_ids.detach().cpu().numpy().astype(np.int64)
    timestep = bank.timesteps.detach().cpu().numpy().astype(np.int64)
    noise = bank.noise_ids.detach().cpu().numpy().astype(np.int64)
    if pair_ids.shape != (int(np.max(source)) + 1,):
        raise ValueError("source_pair_ids shape mismatch")
    pxy = predicted_raw[..., :CABLE_DIM].reshape(-1, DEFAULT_TF, N_BEADS, 2)
    txy = target_raw[..., :CABLE_DIM].reshape(-1, DEFAULT_TF, N_BEADS, 2)
    ratios: List[float] = []
    own_values: List[float] = []
    separation_values: List[float] = []
    below_half: List[float] = []
    for pair in sorted(set(pair_ids.tolist())):
        members = np.flatnonzero(pair_ids == pair)
        if members.size != 2:
            raise ValueError("each pair must contain two source IDs")
        for local_timestep in sorted(set(timestep.tolist())):
            for local_noise in sorted(set(noise.tolist())):
                index = np.flatnonzero(
                    np.isin(source, members)
                    & (timestep == local_timestep)
                    & (noise == local_noise)
                )
                if index.size != 2:
                    raise RuntimeError("paired evaluation bank is incomplete")
                a, b = int(index[0]), int(index[1])
                separation = float(np.sqrt(np.mean(np.square(txy[a] - txy[b]))))
                for row in (a, b):
                    own = float(np.sqrt(np.mean(np.square(pxy[row] - txy[row]))))
                    ratio = own / max(separation, 1.0e-12)
                    own_values.append(own)
                    separation_values.append(separation)
                    ratios.append(ratio)
                    below_half.append(float(ratio < 0.5))
    return {
        "comparison_count": int(len(ratios)),
        "own_ordered_rmse": _stats(own_values),
        "target_branch_separation": _stats(separation_values),
        "own_error_to_branch_separation": _stats(ratios),
        "own_error_below_half_separation_fraction": float(np.mean(below_half)),
    }


def _physical_summary(
    raw: np.ndarray,
    physical_contract: Any,
    physical_evaluator: Callable[[np.ndarray, Any], Mapping[str, Any]],
) -> Dict[str, Any]:
    value = physical_evaluator(raw[None].astype(np.float32), physical_contract)
    output = {
        str(key): item
        for key, item in value.items()
        if key != "sample_valid_mask"
    }
    if "sample_validity_rate" not in output:
        raise ValueError("physical evaluator missing sample_validity_rate")
    return output


def evaluate_prediction_contract(
    *,
    predicted_z: torch.Tensor,
    bank: LabeledTupleBank,
    source_pair_ids: Sequence[int],
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    physical_contract: Any,
    gate: ReconstructionGate = RANDOM_SINGLE_BRANCH_GATE,
    spec: GateSeparationSpec = GateSeparationSpec(),
    physical_evaluator: Callable[[np.ndarray, Any], Mapping[str, Any]] = calibrated_validity,
) -> Dict[str, Any]:
    """Evaluate exact fidelity, cable branch identity and physical validity."""
    spec.validate()
    predicted = predicted_z.detach()
    if predicted.shape != bank.clean_z.shape:
        raise ValueError("predicted_z and bank clean_z shapes differ")
    predicted_raw_tensor = torch_inverse_standardize(
        predicted,
        future_mean,
        future_scale,
    )
    pz = _as_numpy(predicted, name="predicted_z")
    tz = _as_numpy(bank.clean_z, name="target_z")
    pr = _as_numpy(predicted_raw_tensor, name="predicted_raw")
    tr = _as_numpy(bank.clean_raw, name="target_raw")
    active = _as_numpy(active_mask, name="active_mask").astype(bool)
    metrics = target_reconstruction_metrics(
        predicted_z=pz,
        target_z=tz,
        predicted_raw=pr,
        target_raw=tr,
        active_mask=active,
    )
    components = reconstruction_gate_components(metrics, gate)
    branch = paired_branch_audit(
        predicted_z=predicted,
        bank=bank,
        source_pair_ids=source_pair_ids,
        future_mean=future_mean,
        future_scale=future_scale,
    )
    physical = _physical_summary(pr, physical_contract, physical_evaluator)
    topology_raw = nearest_index_metrics(pr, tr)
    inversion = np.asarray(topology_raw["nearest_inversion"], dtype=np.float64)
    unique = np.asarray(
        topology_raw["nearest_unique_fraction"], dtype=np.float64
    )
    if inversion.shape != (predicted.shape[0],) or unique.shape != (predicted.shape[0],):
        raise ValueError("nearest-index metric batch shape mismatch")
    if not np.isfinite(inversion).all() or not np.isfinite(unique).all():
        raise ValueError("nearest-index metric contains non-finite values")
    topology = {
        "source": "frozen_prediction_reference_contract",
        "nearest_inversion": _stats(inversion.tolist()),
        "nearest_unique_fraction": _stats(unique.tolist()),
        "nearest_inversion_p95_threshold": float(
            spec.topology_inversion_p95_max
        ),
        "pass": bool(
            np.percentile(inversion, 95)
            <= float(spec.topology_inversion_p95_max)
        ),
    }
    horizons = {
        str(horizon): _single_horizon_metrics(
            predicted_z=pz,
            target_z=tz,
            predicted_raw=pr,
            target_raw=tr,
            active_mask=active,
            horizon=horizon,
            gate=gate,
        )
        for horizon in range(DEFAULT_TF)
    }
    source = bank.source_ids.detach().cpu().numpy().astype(np.int64)
    timestep = bank.timesteps.detach().cpu().numpy().astype(np.int64)
    noise = bank.noise_ids.detach().cpu().numpy().astype(np.int64)
    by_source = _grouped_reconstruction(
        predicted_z=pz,
        target_z=tz,
        predicted_raw=pr,
        target_raw=tr,
        active_mask=active,
        labels=source,
        gate=gate,
    )
    by_timestep = _grouped_reconstruction(
        predicted_z=pz,
        target_z=tz,
        predicted_raw=pr,
        target_raw=tr,
        active_mask=active,
        labels=timestep,
        gate=gate,
    )
    by_noise = _grouped_reconstruction(
        predicted_z=pz,
        target_z=tz,
        predicted_raw=pr,
        target_raw=tr,
        active_mask=active,
        labels=noise,
        gate=gate,
    )
    group_z = _group_z_metrics(pz, tz, active)
    pair_relative = _pair_relative_error(
        predicted_raw=pr,
        target_raw=tr,
        bank=bank,
        source_pair_ids=source_pair_ids,
    )
    branch_pass = bool(branch.get("pass", False))
    physical_pass = bool(
        float(physical["sample_validity_rate"])
        >= spec.one_step_physical_validity_min
    )
    final_horizon_cable_pass = bool(
        horizons[str(DEFAULT_TF - 1)]["cable_geometry_gate_pass"]
    )
    earlier_horizon_cable_pass = [
        bool(horizons[str(h)]["cable_geometry_gate_pass"])
        for h in range(DEFAULT_TF - 1)
    ]
    return {
        "schema": GATE_SEPARATION_SCHEMA,
        "reconstruction_schema": RECONSTRUCTION_DECOMPOSITION_SCHEMA,
        "row_count": int(predicted.shape[0]),
        "historical_reconstruction_gate": asdict(gate),
        "full_reconstruction_metrics": metrics,
        "full_reconstruction_components": components,
        "state_group_z_metrics": group_z,
        "by_horizon": horizons,
        "by_source": by_source,
        "by_timestep": by_timestep,
        "by_noise": by_noise,
        "pair_relative_error": pair_relative,
        "branch_transport": branch,
        "one_step_physical": physical,
        "ordered_topology": topology,
        "contracts": {
            "historical_exact_reconstruction_pass": bool(
                components["historical_exact_reconstruction_pass"]
            ),
            "branch_transport_pass": branch_pass,
            "one_step_physical_pass": physical_pass,
            "ordered_topology_pass": bool(topology["pass"]),
            "separated_branch_physical_pass": bool(branch_pass and physical_pass),
            "separated_branch_physical_topology_pass": bool(
                branch_pass and physical_pass and topology["pass"]
            ),
            "historical_composite_pass": bool(
                components["historical_exact_reconstruction_pass"] and branch_pass
            ),
            "final_horizon_cable_geometry_pass": final_horizon_cable_pass,
            "all_earlier_horizon_cable_geometry_pass": bool(
                all(earlier_horizon_cable_pass)
            ),
        },
    }


def _pair_row_index(bank: LabeledTupleBank, source_pair_ids: Sequence[int]) -> torch.Tensor:
    pair_ids = np.asarray(source_pair_ids, dtype=np.int64)
    source = bank.source_ids.detach().cpu().numpy().astype(np.int64)
    timestep = bank.timesteps.detach().cpu().numpy().astype(np.int64)
    noise = bank.noise_ids.detach().cpu().numpy().astype(np.int64)
    if pair_ids.shape != (int(np.max(source)) + 1,):
        raise ValueError("source_pair_ids shape mismatch")
    lookup = {
        (int(source[index]), int(timestep[index]), int(noise[index])): int(index)
        for index in range(source.size)
    }
    output: List[int] = []
    for index in range(source.size):
        members = np.flatnonzero(pair_ids == pair_ids[source[index]])
        if members.size != 2:
            raise ValueError("each pair must contain exactly two sources")
        other = int(members[0] if members[1] == source[index] else members[1])
        key = (other, int(timestep[index]), int(noise[index]))
        if key not in lookup:
            raise RuntimeError(f"paired row missing for {key}")
        output.append(lookup[key])
    return torch.tensor(output, device=bank.clean_z.device, dtype=torch.long)


def _raw_to_standardized(
    raw: torch.Tensor,
    *,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    active_mask: torch.Tensor,
) -> torch.Tensor:
    if raw.ndim != 3 or raw.shape[1:] != (DEFAULT_TF, STATE_DIM):
        raise ValueError("raw future must be [N,4,87]")
    scale = future_scale.to(device=raw.device, dtype=raw.dtype)
    mean = future_mean.to(device=raw.device, dtype=raw.dtype)
    active = active_mask.to(device=raw.device, dtype=torch.bool)
    safe = torch.where(torch.abs(scale) > 1.0e-8, scale, torch.ones_like(scale))
    value = (raw - mean) / safe
    return torch.where(active[None], value, torch.zeros_like(value))


def synthetic_prediction_controls(
    *,
    bank: LabeledTupleBank,
    source_pair_ids: Sequence[int],
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    spec: GateSeparationSpec = GateSeparationSpec(),
) -> Dict[str, torch.Tensor]:
    spec.validate()
    paired_index = _pair_row_index(bank, source_pair_ids)
    clean_z = bank.clean_z.detach().clone()
    paired_z = clean_z.index_select(0, paired_index)
    source = bank.source_ids.detach().cpu().numpy().astype(np.int64)
    pair_ids = np.asarray(source_pair_ids, dtype=np.int64)
    first_source_by_pair = {
        pair: int(np.min(np.flatnonzero(pair_ids == pair)))
        for pair in sorted(set(pair_ids.tolist()))
    }
    lookup = {
        (
            int(bank.source_ids[index].item()),
            int(bank.timesteps[index].item()),
            int(bank.noise_ids[index].item()),
        ): int(index)
        for index in range(bank.row_count)
    }
    collapse_index: List[int] = []
    for index in range(bank.row_count):
        pair = int(pair_ids[source[index]])
        first = first_source_by_pair[pair]
        key = (
            first,
            int(bank.timesteps[index].item()),
            int(bank.noise_ids[index].item()),
        )
        collapse_index.append(lookup[key])
    collapse_index_tensor = torch.tensor(
        collapse_index,
        device=clean_z.device,
        dtype=torch.long,
    )

    robot_offset = clean_z.clone()
    robot_active = active_mask.to(device=clean_z.device, dtype=torch.bool).clone()
    robot_active[:, :CABLE_DIM] = False
    robot_offset = torch.where(
        robot_active[None],
        robot_offset + float(spec.robot_proxy_offset_z),
        robot_offset,
    )

    translated_raw = bank.clean_raw.detach().clone()
    translated_xy = translated_raw[..., :CABLE_DIM].reshape(
        -1, DEFAULT_TF, N_BEADS, 2
    )
    translated_xy[..., 0] += float(spec.cable_translation_m)
    translated_raw[..., :CABLE_DIM] = translated_xy.reshape(
        -1, DEFAULT_TF, CABLE_DIM
    )
    translated_z = _raw_to_standardized(
        translated_raw,
        future_mean=future_mean,
        future_scale=future_scale,
        active_mask=active_mask,
    )

    reversed_raw = bank.clean_raw.detach().clone()
    reversed_xy = reversed_raw[..., :CABLE_DIM].reshape(
        -1, DEFAULT_TF, N_BEADS, 2
    )
    reversed_raw[..., :CABLE_DIM] = torch.flip(reversed_xy, dims=(2,)).reshape(
        -1, DEFAULT_TF, CABLE_DIM
    )
    reversed_z = _raw_to_standardized(
        reversed_raw,
        future_mean=future_mean,
        future_scale=future_scale,
        active_mask=active_mask,
    )

    return {
        "exact_oracle": clean_z,
        "branch_swap": paired_z,
        "pair_mean": 0.5 * (clean_z + paired_z),
        "collapse_first_branch": clean_z.index_select(0, collapse_index_tensor),
        "robot_proxy_offset": robot_offset,
        "cable_translation": translated_z,
        "bead_order_reversal": reversed_z,
    }


def evaluate_synthetic_controls(
    *,
    bank: LabeledTupleBank,
    source_pair_ids: Sequence[int],
    active_mask: torch.Tensor,
    future_mean: torch.Tensor,
    future_scale: torch.Tensor,
    physical_contract: Any,
    spec: GateSeparationSpec = GateSeparationSpec(),
    physical_evaluator: Callable[[np.ndarray, Any], Mapping[str, Any]] = calibrated_validity,
) -> Dict[str, Any]:
    predictions = synthetic_prediction_controls(
        bank=bank,
        source_pair_ids=source_pair_ids,
        active_mask=active_mask,
        future_mean=future_mean,
        future_scale=future_scale,
        spec=spec,
    )
    controls = {
        name: evaluate_prediction_contract(
            predicted_z=value,
            bank=bank,
            source_pair_ids=source_pair_ids,
            active_mask=active_mask,
            future_mean=future_mean,
            future_scale=future_scale,
            physical_contract=physical_contract,
            spec=spec,
            physical_evaluator=physical_evaluator,
        )
        for name, value in predictions.items()
    }
    contract = {
        "exact_oracle_pass": bool(
            controls["exact_oracle"]["contracts"][
                "historical_exact_reconstruction_pass"
            ]
            and controls["exact_oracle"]["contracts"]["branch_transport_pass"]
        ),
        "branch_swap_rejected": bool(
            not controls["branch_swap"]["contracts"]["branch_transport_pass"]
        ),
        "pair_mean_rejected": bool(
            not controls["pair_mean"]["contracts"]["branch_transport_pass"]
        ),
        "collapse_rejected": bool(
            not controls["collapse_first_branch"]["contracts"][
                "branch_transport_pass"
            ]
        ),
        "robot_proxy_orthogonality_demonstrated": bool(
            controls["robot_proxy_offset"]["contracts"]["branch_transport_pass"]
            and controls["robot_proxy_offset"]["full_reconstruction_components"][
                "cable_geometry_gate_pass"
            ]
            and not controls["robot_proxy_offset"]["contracts"][
                "historical_exact_reconstruction_pass"
            ]
        ),
        "topology_corruption_rejected": bool(
            not controls["bead_order_reversal"]["contracts"][
                "ordered_topology_pass"
            ]
        ),
    }
    contract["pass"] = bool(all(contract.values()))
    return {
        "schema": SYNTHETIC_CONTROL_SCHEMA,
        "controls": controls,
        "contract": contract,
    }


def compare_reconstruction_to_training_result(
    *,
    observed: Mapping[str, Any],
    training_result: Mapping[str, Any],
    tolerance: float = 2.0e-6,
) -> Dict[str, Any]:
    true_value = training_result.get("evaluations", {}).get("true", {})
    expected = true_value.get("aggregate", {}).get("metrics", {})
    actual = observed.get("full_reconstruction_metrics", {})
    paths = (
        ("z_mse",),
        ("ordered_rmse", "p95"),
        ("segment_relative_error", "p95"),
        ("chain_relative_error", "p95"),
    )

    def read(mapping: Mapping[str, Any], path: Sequence[str]) -> float:
        value: Any = mapping
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                raise ValueError("missing reconstruction metric: " + ".".join(path))
            value = value[key]
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite reconstruction metric")
        return number

    checks: Dict[str, Any] = {}
    passed = True
    for path in paths:
        name = ".".join(path)
        actual_value = read(actual, path)
        expected_value = read(expected, path)
        error = abs(actual_value - expected_value)
        item_pass = bool(error <= float(tolerance))
        checks[name] = {
            "observed": actual_value,
            "expected": expected_value,
            "absolute_error": float(error),
            "tolerance": float(tolerance),
            "pass": item_pass,
        }
        passed = passed and item_pass
    return {
        "schema": HISTORICAL_REPRODUCTION_SCHEMA,
        "checks": checks,
        "pass": bool(passed),
    }


def _failing_group_count(grouped: Mapping[str, Any]) -> int:
    groups = grouped.get("groups", {})
    return int(sum(not bool(value.get("gate_pass")) for value in groups.values()))


def _gate_separation_payload(value: Mapping[str, Any]) -> Mapping[str, Any]:
    separation = value.get("gate_separation")
    if separation is None:
        separation = value
    if not isinstance(separation, Mapping):
        raise ValueError("gate-separation payload must be a mapping")
    return separation


def variant_supported_mechanisms(value: Mapping[str, Any]) -> List[str]:
    mechanisms: List[str] = []
    separation = _gate_separation_payload(value)
    contracts = separation.get("contracts", {})
    components = separation.get("full_reconstruction_components", {})
    group_z = separation.get("state_group_z_metrics", {})
    horizons = separation.get("by_horizon", {})
    if (
        bool(contracts.get("branch_transport_pass"))
        and not bool(contracts.get("historical_exact_reconstruction_pass"))
    ):
        mechanisms.append("exact_reconstruction_vs_branch_identity_separation")
    if (
        not bool(components.get("z_state_gate_pass"))
        and bool(components.get("cable_geometry_gate_pass"))
        and float(group_z.get("robot_proxy_error_fraction", 0.0)) >= 0.50
    ):
        mechanisms.append("robot_proxy_reconstruction_conflation")
    final_pass = bool(
        horizons.get(str(DEFAULT_TF - 1), {}).get("cable_geometry_gate_pass", False)
    )
    earlier = [
        bool(horizons.get(str(index), {}).get("cable_geometry_gate_pass", False))
        for index in range(DEFAULT_TF - 1)
    ]
    if final_pass and not all(earlier):
        mechanisms.append("intermediate_horizon_fidelity_gap")
    if (
        bool(contracts.get("branch_transport_pass"))
        and not bool(components.get("cable_geometry_gate_pass"))
    ):
        mechanisms.append("exact_cable_fidelity_tail_gap")
    if _failing_group_count(separation.get("by_source", {})) > 0:
        mechanisms.append("all_source_tail_failure")
    return mechanisms


def classify_gate_separation(report: Mapping[str, Any]) -> Dict[str, Any]:
    variants = report.get("variants", {})
    if set(variants) != set(DIAGNOSTIC_OBJECTIVE_NAMES):
        return {
            "root_cause": "phase314b_r253_diagnostic_matrix_incomplete",
            "supported_mechanisms": [],
            "next_stage": "repair the immutable three-model gate-separation matrix",
            "train_only_recommendation": None,
        }
    if any(not bool(value.get("training_completed")) for value in variants.values()):
        return {
            "root_cause": "phase314b_r253_training_reproduction_failed",
            "supported_mechanisms": [],
            "next_stage": "repair deterministic r2.5.2 model reproduction",
            "train_only_recommendation": None,
        }
    if any(
        not bool(value.get("r252_one_step_reproduction", {}).get("pass"))
        for value in variants.values()
    ):
        return {
            "root_cause": "phase314b_r253_r252_one_step_reproduction_failed",
            "supported_mechanisms": [],
            "next_stage": "repair one-step reproduction before changing evaluation contracts",
            "train_only_recommendation": None,
        }
    controls = report.get("synthetic_controls", {}).get("contract", {})
    if not bool(controls.get("pass")):
        return {
            "root_cause": "phase314b_r253_gate_separation_controls_failed",
            "supported_mechanisms": [],
            "next_stage": "repair positive/negative gate controls before separating contracts",
            "train_only_recommendation": None,
        }
    if any(
        not bool(
            _gate_separation_payload(value)
            .get("contracts", {})
            .get("branch_transport_pass")
        )
        for value in variants.values()
    ):
        return {
            "root_cause": "phase314b_r253_branch_transport_not_reproduced",
            "supported_mechanisms": [],
            "next_stage": "reconcile r2.5.2 branch-transport reproduction",
            "train_only_recommendation": None,
        }

    mechanisms_by_name = {
        name: variant_supported_mechanisms(value)
        for name, value in variants.items()
    }
    supported: List[str] = []
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        for mechanism in mechanisms_by_name[name]:
            if mechanism not in supported:
                supported.append(mechanism)

    count = lambda mechanism: sum(  # noqa: E731
        mechanism in mechanisms_by_name[name]
        for name in DIAGNOSTIC_OBJECTIVE_NAMES
    )
    if count("robot_proxy_reconstruction_conflation") >= 2:
        root = "phase314b_r253_robot_proxy_reconstruction_conflation_supported"
        next_stage = (
            "audit a separate robot-proxy trajectory-fidelity objective under the fixed "
            "cable branch-transport contract"
        )
    elif count("intermediate_horizon_fidelity_gap") >= 2:
        root = "phase314b_r253_intermediate_horizon_fidelity_gap_supported"
        next_stage = (
            "repair intermediate-horizon trajectory fidelity without changing the "
            "established cable branch-transport gate"
        )
    elif count("exact_cable_fidelity_tail_gap") >= 2:
        root = "phase314b_r253_exact_cable_fidelity_tail_gap_supported"
        next_stage = (
            "audit cable trajectory-fidelity tails separately from branch identity"
        )
    elif count("all_source_tail_failure") >= 2:
        root = "phase314b_r253_all_source_reconstruction_tail_failure_supported"
        next_stage = "localize the persistent source-level reconstruction tail"
    elif count("exact_reconstruction_vs_branch_identity_separation") >= 2:
        root = "phase314b_r253_gate_separation_contract_supported"
        next_stage = (
            "retain branch transport as an independent research gate and separately "
            "repair exact state-trajectory fidelity"
        )
    else:
        root = "phase314b_r253_reconstruction_failure_unattributed"
        next_stage = "expand train-only reconstruction diagnostics without changing training"
    return {
        "root_cause": root,
        "supported_mechanisms": supported,
        "mechanisms_by_variant": mechanisms_by_name,
        "next_stage": next_stage,
        "train_only_recommendation": None,
    }


def strip_runtime_objects(value: Any) -> Any:
    if torch.is_tensor(value):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        raise TypeError("runtime tensor must not be serialized")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {
            str(key): strip_runtime_objects(item)
            for key, item in value.items()
            if not str(key).startswith("_")
        }
    if isinstance(value, tuple):
        return [strip_runtime_objects(item) for item in value]
    if isinstance(value, list):
        return [strip_runtime_objects(item) for item in value]
    return value

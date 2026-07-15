"""Phase3.14b-r2.5.6 Stage D.2 cable-XY collapse provenance audit.

Stage D.1 showed that a calibration ground-truth segment with an XY length
ratio near 0.00125 drives a directionally coupled segment gate.  The state-v3
cable target stores only ordered bead XY, while the immutable raw episode stores
the same ordered beads as XYZ.

This stage performs a source-only audit.  It does not train or replay a model.
It reconstructs every Stage-A train/full-horizon target directly from its
source pickle, verifies the exact window-to-frame mapping, compares raw XY to
the cached float32 target byte-for-byte, and determines whether severe XY
collapse is:

* a projection effect (XYZ length remains normal),
* a true three-dimensional segment collapse,
* a cache/extraction mismatch, or
* a bead-order/provenance failure.

The audit also corrects an important naming ambiguity: ``row_index`` in the
Stage-A contract is the 4256-row window-contract index.  It is not a raw pickle
frame index.  The raw frame index for target horizon ``h`` is exactly::

    raw_info_index = window_t + 1 + h

for the full-horizon rows audited here.

No gate is selected, no threshold is changed, DeformableRavens is not executed,
and no cache, NPZ, tensor, model, image, or video is written.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import subprocess
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from ccda_phase3 import phase314b_r256_stageb_cable_diffusion as stageb
from ccda_phase3 import phase314b_r256_staged_segment_recalibration as staged
from ccda_phase3 import phase314b_r256_staged1_asymmetric_gate_audit as staged1

PHASE = "Phase3.14b-r2.5.6 Stage D.2"
PHASE_ID = "phase314b_r256_staged2"
BASE_EVIDENCE_COMMIT = "3acfe62dcb37173d4a5b256ecd01acac6e18460c"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"

FORMAL_RAW_ROOT = "data/phase3_state_v2_slack"
STAGE_D_GATE = "reports/phase3_14b_r256_staged_segment_gate_contract.json"

STAGED1_SOURCE = (
    "ccda_phase3/phase314b_r256_staged1_asymmetric_gate_audit.py"
)
STAGED1_TEST_GATE = (
    "reports/phase3_14b_r256_staged1_test_gate_summary.json"
)
STAGED1_GATE_AUDIT = "reports/phase3_14b_r256_staged1_gate_audit.json"
STAGED1_WORKER_EVIDENCE = (
    "reports/phase3_14b_r256_staged1_worker_evidence.json"
)
STAGED1_SUMMARY = "reports/phase3_14b_r256_staged1_summary.json"
STAGED1_REPORT = "reports/phase3_14b_r256_staged1_report.md"

BASE_BOUND_FILES = (
    STAGED1_SOURCE,
    STAGED1_TEST_GATE,
    STAGED1_GATE_AUDIT,
    STAGED1_WORKER_EVIDENCE,
    STAGED1_SUMMARY,
    STAGED1_REPORT,
    STAGE_D_GATE,
)

MAIN_SOURCE_FILES = (
    "ccda_phase3/data_io.py",
    "ccda_phase3/schema_v3.py",
    "ccda_phase3/phase314b_r255_stageb_dataset.py",
    "ccda_phase3/phase314b_r256_stagea_contract.py",
)

SUBMODULE_SOURCE_FILES = (
    "ravens/tasks/ccda_slack_cable_v2.py",
    "ravens/tasks/defs_cables.py",
)

TRAIN_REQUIRED_KEYS = (
    "diffusion_target_cable",
    "future_valid_mask",
    "condition_name",
    "split_name",
    "visible_seed",
    "pair_group",
    "pair_key",
    "episode_group_key",
    "window_t",
    "source_row_index",
)

CONTRACT_REQUIRED_KEYS = (
    "row_index",
    "diffusion_target_cable",
    "future_valid_mask",
    "condition_name",
    "split_name",
    "visible_seed",
    "pair_group",
    "pair_key",
    "episode_group_key",
    "source_file",
    "source_pickle_sha256",
    "episode_index",
    "window_t",
)

N_BEADS = stageb.BEADS
FUTURE_STEPS = stageb.FUTURE_STEPS
CABLE_DIM = stageb.CABLE_DIM
SEGMENT_COUNT = N_BEADS - 1
EXPECTED_TRAIN_FULL_ROWS = 1000
NOMINAL_CABLE_RADIUS = 0.005
NOMINAL_SEGMENT_LENGTH = (
    2.0 * NOMINAL_CABLE_RADIUS * math.sqrt(2.0)
)


class CollapseProvenanceError(RuntimeError):
    """Raised when source provenance cannot be reconstructed exactly."""


@dataclass(frozen=True)
class ProvenanceAuditSpec:
    """Pre-registered source-audit and classification thresholds."""

    severe_xy_ratio: float = 0.25
    moderate_xy_ratio: float = 0.50
    mild_xy_ratio: float = 0.75

    xyz_collapsed_ratio: float = 0.25
    xyz_normal_ratio: float = 0.75

    projection_dominance_min: float = 0.90
    true_3d_fraction_max_for_projection: float = 0.01
    true_3d_fraction_min: float = 0.10

    cache_float32_exact_required: bool = True
    bead_count: int = 24
    raw_position_dim: int = 3
    orientation_dim: int = 4

    top_event_records: int = 64
    top_source_records: int = 32
    timeline_neighbor_radius: int = 1

    def validate(self) -> None:
        if not (
            0.0
            < self.severe_xy_ratio
            < self.moderate_xy_ratio
            < self.mild_xy_ratio
            < 1.0
        ):
            raise ValueError("XY collapse thresholds are invalid")
        if not (
            0.0
            < self.xyz_collapsed_ratio
            < self.xyz_normal_ratio
            <= 1.0
        ):
            raise ValueError("XYZ classification thresholds are invalid")
        if not 0.0 < self.projection_dominance_min <= 1.0:
            raise ValueError("projection dominance threshold is invalid")
        if not 0.0 <= self.true_3d_fraction_max_for_projection < 1.0:
            raise ValueError("true-3D projection allowance is invalid")
        if not 0.0 < self.true_3d_fraction_min <= 1.0:
            raise ValueError("true-3D fraction threshold is invalid")
        if self.bead_count != N_BEADS:
            raise ValueError("bead count differs from frozen schema")
        if self.raw_position_dim != 3 or self.orientation_dim != 4:
            raise ValueError("raw bead-state dimensions changed")
        if self.top_event_records <= 0 or self.top_source_records <= 0:
            raise ValueError("record limits must be positive")
        if self.timeline_neighbor_radius < 0:
            raise ValueError("timeline neighbor radius is negative")


@dataclass(frozen=True)
class RawFrame:
    positions_xyz: np.ndarray
    bead_ids: np.ndarray
    orientations_xyzw: np.ndarray
    visible_seed: Optional[int]
    pair_group: str
    condition: str
    task_name: str

    def validate(self, spec: ProvenanceAuditSpec) -> None:
        if self.positions_xyz.shape != (
            spec.bead_count,
            spec.raw_position_dim,
        ):
            raise CollapseProvenanceError(
                f"raw bead_positions shape changed: "
                f"{self.positions_xyz.shape}"
            )
        if self.bead_ids.shape != (spec.bead_count,):
            raise CollapseProvenanceError(
                f"raw bead_ids shape changed: {self.bead_ids.shape}"
            )
        if self.orientations_xyzw.shape != (
            spec.bead_count,
            spec.orientation_dim,
        ):
            raise CollapseProvenanceError(
                "raw bead_orientations shape changed: "
                f"{self.orientations_xyzw.shape}"
            )
        if not np.all(np.isfinite(self.positions_xyz)):
            raise CollapseProvenanceError(
                "raw bead positions contain NaN or Inf"
            )
        if not np.all(np.isfinite(self.orientations_xyzw)):
            raise CollapseProvenanceError(
                "raw bead orientations contain NaN or Inf"
            )
        if len(set(self.bead_ids.tolist())) != spec.bead_count:
            raise CollapseProvenanceError(
                "raw bead IDs are not unique"
            )


@dataclass(frozen=True)
class LoadedEpisode:
    source_file: str
    source_sha256: str
    frames: Tuple[RawFrame, ...]
    action_count: int
    manifest: Mapping[str, Any]

    def validate(self) -> None:
        if len(self.frames) != self.action_count + 1:
            raise CollapseProvenanceError(
                f"source episode frame/action mismatch for "
                f"{self.source_file}: {len(self.frames)} vs "
                f"{self.action_count}"
            )
        if not self.frames:
            raise CollapseProvenanceError(
                f"source episode has no frames: {self.source_file}"
            )
        reference_ids = self.frames[0].bead_ids
        for index, frame in enumerate(self.frames):
            if not np.array_equal(frame.bead_ids, reference_ids):
                raise CollapseProvenanceError(
                    "bead ID order changed within episode "
                    f"{self.source_file} at frame {index}"
                )


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(root),
        text=True,
    ).strip()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(list(array.shape)).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def sha256_strings(values: Sequence[Any]) -> str:
    return sha256_bytes(
        "\n".join(str(value) for value in values).encode("utf-8")
    )


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "fc" and not np.all(np.isfinite(value)):
            raise ValueError("non-finite array cannot be serialized")
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {
            str(key): jsonable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("non-finite scalar cannot be serialized")
        return value
    raise TypeError(f"unsupported JSON value: {type(value)!r}")


def stable_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            jsonable(dict(payload)),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def atomic_write_once(
    path: Path,
    payload: bytes,
    *,
    mode: int = 0o644,
) -> None:
    target = Path(path)
    if target.exists():
        raise FileExistsError(
            f"refusing to overwrite write-once output: {target}"
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, target)
        directory_fd = os.open(str(target.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_json(path: Path) -> Dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CollapseProvenanceError(
            f"JSON root is not an object: {path}"
        )
    return value


def _safe_stats(value: np.ndarray) -> Dict[str, float]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "p001": 0.0,
            "p01": 0.0,
            "p05": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "minimum": 0.0,
            "maximum": 0.0,
        }
    if not np.all(np.isfinite(array)):
        raise CollapseProvenanceError(
            "statistics input contains NaN or Inf"
        )
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p001": float(np.percentile(array, 0.1)),
        "p01": float(np.percentile(array, 1.0)),
        "p05": float(np.percentile(array, 5.0)),
        "p50": float(np.percentile(array, 50.0)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def assert_base_file_bound(root: Path, relative: str) -> str:
    path = Path(root) / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    committed = subprocess.check_output(
        ["git", "show", f"{BASE_EVIDENCE_COMMIT}:{relative}"],
        cwd=str(root),
    )
    observed = path.read_bytes()
    if committed != observed:
        raise CollapseProvenanceError(
            f"base-bound Stage-D.1 file changed: {relative}"
        )
    return sha256_bytes(observed)


def validate_immutable_inputs(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    bound = {
        relative: assert_base_file_bound(repository_root, relative)
        for relative in BASE_BOUND_FILES
    }
    summary = load_json(repository_root / STAGED1_SUMMARY)
    if summary.get("verdict") != "PASS":
        raise CollapseProvenanceError(
            "Stage-D.1 verdict is not PASS"
        )
    if summary.get("scientific_status") != "BLOCKED":
        raise CollapseProvenanceError(
            "Stage-D.1 scientific status changed"
        )
    if summary.get("root_cause") != (
        "phase314b_r256_staged1_lower_tail_ground_truth_"
        "collapse_drives_directional_gate_coupling"
    ):
        raise CollapseProvenanceError(
            "Stage-D.1 root cause changed"
        )
    if summary.get("required_next_path") != (
        "AUDIT_CABLE_XY_SEGMENT_COLLAPSE_PROVENANCE_"
        "BEFORE_GATE_OR_BRANCH_REPAIR"
    ):
        raise CollapseProvenanceError(
            "Stage-D.1 next path changed"
        )
    if summary.get("gate_selected") is not False:
        raise CollapseProvenanceError(
            "Stage-D.1 unexpectedly selected a gate"
        )
    if summary.get("threshold_changed") is not False:
        raise CollapseProvenanceError(
            "Stage-D.1 unexpectedly changed a threshold"
        )
    if summary.get("train_only_recommendation") is not None:
        raise CollapseProvenanceError(
            "Stage-D.1 selected a train-only recommendation"
        )
    if summary.get("selected_configuration") is not None:
        raise CollapseProvenanceError(
            "Stage-D.1 selected a configuration"
        )

    evidence = load_json(repository_root / STAGED1_WORKER_EVIDENCE)
    if evidence.get("workers_exact") is not True:
        raise CollapseProvenanceError(
            "Stage-D.1 workers were not exact"
        )

    audit = load_json(repository_root / STAGED1_GATE_AUDIT)
    classification = audit.get("classification")
    if not isinstance(classification, dict):
        raise CollapseProvenanceError(
            "Stage-D.1 classification is missing"
        )
    if classification.get("primary_failure_locus") != (
        "ground_truth_segment_collapse_provenance"
    ):
        raise CollapseProvenanceError(
            "Stage-D.1 primary failure locus changed"
        )
    driver = audit.get("threshold_driver_attribution")
    if not isinstance(driver, dict):
        raise CollapseProvenanceError(
            "Stage-D.1 threshold driver is missing"
        )
    lower_drivers = driver.get("lower_threshold_drivers")
    if not isinstance(lower_drivers, list) or not lower_drivers:
        raise CollapseProvenanceError(
            "Stage-D.1 lower driver records are missing"
        )

    upstream = staged1.validate_immutable_inputs(repository_root)
    return {
        "base_commit": BASE_EVIDENCE_COMMIT,
        "base_file_sha256": bound,
        "staged1_summary": summary,
        "staged1_worker_evidence": evidence,
        "staged1_gate_audit": audit,
        "upstream_immutable": upstream,
    }


def source_logic_audit(root: Path) -> Dict[str, Any]:
    repository_root = Path(root).resolve()
    submodule = repository_root / "external/deformable-ravens"
    main_text = {
        relative: (repository_root / relative).read_text(
            encoding="utf-8"
        )
        for relative in MAIN_SOURCE_FILES
    }
    submodule_text = {
        relative: (submodule / relative).read_text(
            encoding="utf-8"
        )
        for relative in SUBMODULE_SOURCE_FILES
    }

    data_io = main_text["ccda_phase3/data_io.py"]
    schema = main_text["ccda_phase3/schema_v3.py"]
    dataset = main_text[
        "ccda_phase3/phase314b_r255_stageb_dataset.py"
    ]
    contract = main_text[
        "ccda_phase3/phase314b_r256_stagea_contract.py"
    ]
    task = submodule_text[
        "ravens/tasks/ccda_slack_cable_v2.py"
    ]
    cable = submodule_text["ravens/tasks/defs_cables.py"]

    checks = {
        "raw_task_enumerates_cable_bead_ids_in_order": (
            "for index, bead in enumerate(self.cable_bead_IDs)"
            in task
        ),
        "raw_task_stores_xyz_positions": (
            '"bead_positions": [item["position"] for item in beads]'
            in task
        ),
        "raw_task_stores_bead_ids_in_same_list_order": (
            '"bead_ids": [item["id"] for item in beads]'
            in task
        ),
        "cable_builder_appends_ids_in_creation_order": (
            "self.cable_bead_IDs.append(part_id)" in cable
        ),
        "cable_builder_constrains_previous_to_current": (
            "parentBodyUniqueId=env.objects[-1]" in cable
            and "childBodyUniqueId=part_id" in cable
        ),
        "main_extractor_preserves_order_and_drops_only_z": (
            "result = array[:, :2].astype(np.float32, copy=True)"
            in data_io
        ),
        "stageb_materializes_extracted_xy_without_reordering": (
            "cable_xy = extract_bead_xy(info)" in dataset
            and "state_v3_from_components(cable_xy" in dataset
        ),
        "schema_flattens_ordered_xy_directly": (
            "cable.reshape(-1)" in schema
        ),
        "future_target_starts_at_current_plus_one": (
            "start = int(current_index) + 1" in schema
        ),
        "stagea_target_is_exact_cable_slice": (
            "cable_future = full_future[..., :CABLE_DIM].copy()"
            in contract
        ),
        "stagea_train_view_source_index_is_window_row_index": (
            'train_view["source_row_index"] = np.flatnonzero('
            in contract
        ),
    }
    return {
        "checks": checks,
        "all_confirmed": bool(all(checks.values())),
        "main_source_sha256": {
            relative: sha256_file(repository_root / relative)
            for relative in MAIN_SOURCE_FILES
        },
        "submodule_source_sha256": {
            relative: sha256_file(submodule / relative)
            for relative in SUBMODULE_SOURCE_FILES
        },
        "nominal_cable_radius": NOMINAL_CABLE_RADIUS,
        "nominal_center_distance":
            NOMINAL_SEGMENT_LENGTH,
        "ordered_adjacency_contract": (
            "cable_bead_IDs creation order; each new bead is "
            "point-to-point constrained to the previous bead"
        ),
        "state_v3_projection_contract": (
            "raw ordered XYZ -> array[:, :2] -> ordered XY float32"
        ),
    }


def _mapping_extras(info: Any) -> Mapping[str, Any]:
    if not isinstance(info, Mapping):
        raise CollapseProvenanceError(
            "raw episode info is not a mapping"
        )
    extras = info.get("extras")
    if not isinstance(extras, Mapping):
        raise CollapseProvenanceError(
            "raw episode info has no extras mapping"
        )
    return extras


def _optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def extract_raw_frame(
    info: Any,
    *,
    spec: ProvenanceAuditSpec,
) -> RawFrame:
    extras = _mapping_extras(info)
    positions = np.asarray(
        extras.get("bead_positions"),
        dtype=np.float64,
    )
    bead_ids = np.asarray(
        extras.get("bead_ids"),
        dtype=np.int64,
    )
    orientations = np.asarray(
        extras.get("bead_orientations"),
        dtype=np.float64,
    )
    frame = RawFrame(
        positions_xyz=positions,
        bead_ids=bead_ids,
        orientations_xyzw=orientations,
        visible_seed=_optional_int(
            extras.get("ccda_visible_seed")
        ),
        pair_group=str(extras.get("ccda_pair_group", "")),
        condition=str(extras.get("hidden_condition", "")),
        task_name=str(extras.get("ccda_task", "")),
    )
    frame.validate(spec)
    return frame


def load_source_episode(
    path: Path,
    *,
    source_file: str,
    expected_sha256: str,
    spec: ProvenanceAuditSpec,
) -> LoadedEpisode:
    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    observed_sha = sha256_file(source_path)
    if observed_sha != str(expected_sha256):
        raise CollapseProvenanceError(
            f"source pickle SHA changed: {source_file}: "
            f"{observed_sha}"
        )
    with source_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):
        raise CollapseProvenanceError(
            f"source pickle is not a mapping: {source_file}"
        )
    infos = list(payload.get("infos", []))
    actions = list(payload.get("actions", []))
    last_info = payload.get("last_info")
    if not infos or last_info is None:
        raise CollapseProvenanceError(
            f"source episode is incomplete: {source_file}"
        )
    if len(infos) != len(actions):
        raise CollapseProvenanceError(
            f"source infos/actions differ: {source_file}: "
            f"{len(infos)} vs {len(actions)}"
        )
    frames = tuple(
        extract_raw_frame(info, spec=spec)
        for info in infos + [last_info]
    )
    manifest = payload.get("manifest", {})
    if not isinstance(manifest, Mapping):
        raise CollapseProvenanceError(
            f"source manifest is not a mapping: {source_file}"
        )
    episode = LoadedEpisode(
        source_file=str(source_file),
        source_sha256=observed_sha,
        frames=frames,
        action_count=len(actions),
        manifest=dict(manifest),
    )
    episode.validate()
    return episode


class EpisodeStore:
    """SHA-verified immutable source-pickle cache."""

    def __init__(
        self,
        *,
        repository_root: Path,
        formal_root: Path,
        spec: ProvenanceAuditSpec,
    ) -> None:
        self.repository_root = Path(repository_root).resolve()
        self.formal_root = Path(formal_root).resolve()
        self.spec = spec
        self._episodes: Dict[str, LoadedEpisode] = {}
        self._expected_sha: Dict[str, str] = {}

    def get(
        self,
        source_file: str,
        expected_sha256: str,
    ) -> LoadedEpisode:
        relative = Path(str(source_file))
        if relative.is_absolute() or ".." in relative.parts:
            raise CollapseProvenanceError(
                f"source path is not canonical relative: {source_file}"
            )
        path = (self.formal_root / relative).resolve()
        try:
            path.relative_to(self.formal_root)
        except ValueError as exc:
            raise CollapseProvenanceError(
                f"source path escapes formal root: {source_file}"
            ) from exc

        previous = self._expected_sha.get(str(source_file))
        if previous is not None and previous != str(expected_sha256):
            raise CollapseProvenanceError(
                f"one source file has multiple expected SHAs: "
                f"{source_file}"
            )
        self._expected_sha[str(source_file)] = str(expected_sha256)

        if str(source_file) not in self._episodes:
            self._episodes[str(source_file)] = load_source_episode(
                path,
                source_file=str(source_file),
                expected_sha256=str(expected_sha256),
                spec=self.spec,
            )
        return self._episodes[str(source_file)]

    @property
    def source_count(self) -> int:
        return len(self._episodes)

    def source_sha256(self) -> Dict[str, str]:
        return {
            key: self._episodes[key].source_sha256
            for key in sorted(self._episodes)
        }


def validate_contract_mapping(
    train_view: Mapping[str, np.ndarray],
    full_contract: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    source_index = np.asarray(
        train_view["source_row_index"],
        dtype=np.int64,
    )
    if source_index.shape != (EXPECTED_TRAIN_FULL_ROWS,):
        raise CollapseProvenanceError(
            f"source_row_index shape changed: "
            f"{source_index.shape}"
        )
    if np.any(source_index < 0):
        raise CollapseProvenanceError(
            "source_row_index contains a negative value"
        )
    if len(set(source_index.tolist())) != source_index.size:
        raise CollapseProvenanceError(
            "source_row_index is not unique"
        )

    full_rows = np.asarray(
        full_contract["row_index"],
        dtype=np.int64,
    )
    if not np.array_equal(
        full_rows,
        np.arange(full_rows.size, dtype=np.int64),
    ):
        raise CollapseProvenanceError(
            "full contract row_index is not identity"
        )
    if int(np.max(source_index)) >= full_rows.size:
        raise CollapseProvenanceError(
            "source_row_index exceeds full contract"
        )

    exact_keys = (
        "diffusion_target_cable",
        "future_valid_mask",
        "condition_name",
        "split_name",
        "visible_seed",
        "pair_group",
        "pair_key",
        "episode_group_key",
        "window_t",
    )
    exact = {}
    for key in exact_keys:
        observed = np.asarray(train_view[key])
        expected = np.asarray(full_contract[key])[source_index]
        exact[key] = bool(np.array_equal(observed, expected))
        if not exact[key]:
            raise CollapseProvenanceError(
                f"train/full-contract mapping differs: {key}"
            )
    if not np.all(
        np.asarray(train_view["future_valid_mask"], dtype=np.bool_)
    ):
        raise CollapseProvenanceError(
            "Stage-D.2 requires full-horizon rows only"
        )
    if set(
        np.asarray(train_view["split_name"]).astype(str).tolist()
    ) != {"train"}:
        raise CollapseProvenanceError(
            "train/full-horizon view contains non-train rows"
        )

    return {
        "train_view_rows": int(source_index.size),
        "full_contract_rows": int(full_rows.size),
        "source_row_index_unique": True,
        "source_row_index_semantics": (
            "index into the 4256-row window contract; not a raw "
            "pickle frame index"
        ),
        "raw_info_index_formula": "window_t + 1 + future_horizon",
        "exact_key_mapping": exact,
        "mapping_exact": bool(all(exact.values())),
        "source_row_index_sha256": sha256_array(source_index),
    }


def reconstruct_raw_targets(
    *,
    repository_root: Path,
    train_view: Mapping[str, np.ndarray],
    full_contract: Mapping[str, np.ndarray],
    spec: ProvenanceAuditSpec,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any], EpisodeStore]:
    source_index = np.asarray(
        train_view["source_row_index"],
        dtype=np.int64,
    )
    target = np.asarray(
        train_view["diffusion_target_cable"],
        dtype=np.float32,
    )
    if target.shape != (
        EXPECTED_TRAIN_FULL_ROWS,
        FUTURE_STEPS,
        CABLE_DIM,
    ):
        raise CollapseProvenanceError(
            f"train cable target shape changed: {target.shape}"
        )

    source_file = np.asarray(
        full_contract["source_file"]
    ).astype(str)[source_index]
    source_sha = np.asarray(
        full_contract["source_pickle_sha256"]
    ).astype(str)[source_index]
    episode_index = np.asarray(
        full_contract["episode_index"],
        dtype=np.int64,
    )[source_index]
    window_t = np.asarray(
        train_view["window_t"],
        dtype=np.int64,
    )
    condition = np.asarray(
        train_view["condition_name"]
    ).astype(str)
    visible_seed = np.asarray(
        train_view["visible_seed"],
        dtype=np.int64,
    )
    pair_group = np.asarray(
        train_view["pair_group"]
    ).astype(str)

    store = EpisodeStore(
        repository_root=repository_root,
        formal_root=repository_root / FORMAL_RAW_ROOT,
        spec=spec,
    )

    xyz = np.empty(
        (
            EXPECTED_TRAIN_FULL_ROWS,
            FUTURE_STEPS,
            N_BEADS,
            3,
        ),
        dtype=np.float64,
    )
    bead_ids = np.empty(
        (
            EXPECTED_TRAIN_FULL_ROWS,
            FUTURE_STEPS,
            N_BEADS,
        ),
        dtype=np.int64,
    )
    raw_info_index = np.empty(
        (
            EXPECTED_TRAIN_FULL_ROWS,
            FUTURE_STEPS,
        ),
        dtype=np.int64,
    )
    xy_exact = np.ones(
        (
            EXPECTED_TRAIN_FULL_ROWS,
            FUTURE_STEPS,
        ),
        dtype=np.bool_,
    )
    xy_max_abs = np.zeros_like(
        raw_info_index,
        dtype=np.float64,
    )
    metadata_mismatch: List[Dict[str, Any]] = []
    task_names: Counter = Counter()
    orientation_norms: List[float] = []

    for row in range(EXPECTED_TRAIN_FULL_ROWS):
        episode = store.get(
            source_file[row],
            source_sha[row],
        )
        manifest_seed = _optional_int(
            episode.manifest.get("visible_seed")
        )
        manifest_pair = str(
            episode.manifest.get("pair_group", "")
        )
        if (
            manifest_seed is not None
            and manifest_seed != int(visible_seed[row])
        ):
            metadata_mismatch.append(
                {
                    "row": row,
                    "kind": "manifest_visible_seed",
                    "expected": int(visible_seed[row]),
                    "observed": manifest_seed,
                    "source_file": source_file[row],
                }
            )
        if (
            manifest_pair
            and manifest_pair != str(pair_group[row])
        ):
            metadata_mismatch.append(
                {
                    "row": row,
                    "kind": "manifest_pair_group",
                    "expected": str(pair_group[row]),
                    "observed": manifest_pair,
                    "source_file": source_file[row],
                }
            )

        for horizon in range(FUTURE_STEPS):
            info_index = int(window_t[row]) + 1 + horizon
            if not 0 <= info_index < len(episode.frames):
                raise CollapseProvenanceError(
                    "raw target frame is outside episode: "
                    f"row={row}, source={source_file[row]}, "
                    f"window_t={window_t[row]}, horizon={horizon}, "
                    f"info_index={info_index}, "
                    f"frame_count={len(episode.frames)}"
                )
            frame = episode.frames[info_index]
            raw_info_index[row, horizon] = info_index
            xyz[row, horizon] = frame.positions_xyz
            bead_ids[row, horizon] = frame.bead_ids
            orientation_norms.extend(
                np.linalg.norm(
                    frame.orientations_xyzw,
                    axis=1,
                ).astype(np.float64).tolist()
            )
            task_names[frame.task_name] += 1

            raw_xy32 = frame.positions_xyz[:, :2].astype(
                np.float32
            )
            cached_xy32 = target[row, horizon].reshape(
                N_BEADS,
                2,
            )
            exact = bool(np.array_equal(raw_xy32, cached_xy32))
            xy_exact[row, horizon] = exact
            xy_max_abs[row, horizon] = float(
                np.max(
                    np.abs(
                        raw_xy32.astype(np.float64)
                        - cached_xy32.astype(np.float64)
                    )
                )
            )
            if frame.visible_seed is not None and (
                frame.visible_seed != int(visible_seed[row])
            ):
                metadata_mismatch.append(
                    {
                        "row": row,
                        "horizon": horizon,
                        "raw_info_index": info_index,
                        "kind": "frame_visible_seed",
                        "expected": int(visible_seed[row]),
                        "observed": frame.visible_seed,
                        "source_file": source_file[row],
                    }
                )
            if frame.pair_group and (
                frame.pair_group != str(pair_group[row])
            ):
                metadata_mismatch.append(
                    {
                        "row": row,
                        "horizon": horizon,
                        "raw_info_index": info_index,
                        "kind": "frame_pair_group",
                        "expected": str(pair_group[row]),
                        "observed": frame.pair_group,
                        "source_file": source_file[row],
                    }
                )
            if frame.condition and (
                frame.condition != str(condition[row])
            ):
                metadata_mismatch.append(
                    {
                        "row": row,
                        "horizon": horizon,
                        "raw_info_index": info_index,
                        "kind": "frame_condition",
                        "expected": str(condition[row]),
                        "observed": frame.condition,
                        "source_file": source_file[row],
                    }
                )

    if spec.cache_float32_exact_required and not np.all(xy_exact):
        bad = np.argwhere(~xy_exact)
        raise CollapseProvenanceError(
            f"raw XY differs from cached target at "
            f"{bad[:10].tolist()}"
        )
    if metadata_mismatch:
        raise CollapseProvenanceError(
            "source metadata differs from immutable contract: "
            f"{metadata_mismatch[:10]}"
        )

    first_id = bead_ids[:, :1, :]
    id_stable_across_target_horizons = bool(
        np.array_equal(
            bead_ids,
            np.repeat(
                first_id,
                FUTURE_STEPS,
                axis=1,
            ),
        )
    )
    if not id_stable_across_target_horizons:
        raise CollapseProvenanceError(
            "bead ID order changed across target horizons"
        )

    raw = {
        "positions_xyz": xyz,
        "bead_ids": bead_ids,
        "raw_info_index": raw_info_index,
        "source_file": source_file,
        "source_pickle_sha256": source_sha,
        "episode_index": episode_index,
        "window_contract_row_index": source_index,
        "window_t": window_t,
        "condition_name": condition,
        "visible_seed": visible_seed,
        "pair_group": pair_group,
        "pair_key": np.asarray(
            train_view["pair_key"]
        ).astype(str),
        "episode_group_key": np.asarray(
            train_view["episode_group_key"]
        ).astype(str),
    }
    audit = {
        "source_episode_count": store.source_count,
        "raw_target_shape": list(xyz.shape),
        "raw_target_xyz_sha256": sha256_array(xyz),
        "raw_bead_id_sha256": sha256_array(bead_ids),
        "raw_info_index_sha256": sha256_array(raw_info_index),
        "raw_xy_float32_exact_rate": float(np.mean(xy_exact)),
        "raw_xy_float32_exact_count": int(np.sum(xy_exact)),
        "raw_xy_float32_comparison_count": int(xy_exact.size),
        "raw_xy_max_abs_difference": float(
            np.max(xy_max_abs)
        ),
        "bead_id_order_stable_across_target_horizons":
            id_stable_across_target_horizons,
        "orientation_norm": _safe_stats(
            np.asarray(orientation_norms, dtype=np.float64)
        ),
        "task_name_counts": {
            key: int(task_names[key])
            for key in sorted(task_names)
        },
        "metadata_mismatch_count": len(metadata_mismatch),
        "source_pickle_sha256": store.source_sha256(),
    }
    return raw, audit, store


def segment_lengths_xyz(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 4 or array.shape[2:] != (
        N_BEADS,
        3,
    ):
        raise ValueError(
            f"XYZ cable must be [N,4,24,3], got "
            f"{array.shape}"
        )
    delta = array[:, :, 1:, :] - array[:, :, :-1, :]
    length = np.linalg.norm(delta, axis=-1)
    if not np.all(np.isfinite(length)):
        raise CollapseProvenanceError(
            "XYZ segment length contains NaN or Inf"
        )
    if np.any(length <= 0.0):
        # Preserve the evidence as a hard source defect instead of
        # taking log(0) or silently flooring it.
        raise CollapseProvenanceError(
            "raw source contains a non-positive XYZ segment length"
        )
    return length


def segment_components(
    value: np.ndarray,
) -> Dict[str, np.ndarray]:
    array = np.asarray(value, dtype=np.float64)
    delta = array[:, :, 1:, :] - array[:, :, :-1, :]
    xyz = np.linalg.norm(delta, axis=-1)

    # Match the frozen state-v3 / Stage-D XY arithmetic exactly:
    # raw XYZ is first converted to float32, Z is dropped, the ordered
    # difference is computed in float32, and only then is the norm
    # accumulated in float64.
    xy_positions32 = array[..., :2].astype(np.float32)
    delta_xy32 = (
        xy_positions32[:, :, 1:, :]
        - xy_positions32[:, :, :-1, :]
    )
    xy = np.linalg.norm(
        delta_xy32.astype(np.float64),
        axis=-1,
    )
    dz = np.abs(delta[..., 2])
    projection = np.divide(
        xy,
        xyz,
        out=np.zeros_like(xy),
        where=xyz > 0.0,
    )
    vertical = np.divide(
        dz,
        xyz,
        out=np.zeros_like(dz),
        where=xyz > 0.0,
    )
    return {
        "delta_xyz": delta,
        "xyz_length": xyz,
        "xy_length": xy,
        "abs_dz": dz,
        "xy_over_xyz": projection,
        "abs_dz_over_xyz": vertical,
    }


def fit_xyz_reference(
    xyz_lengths: np.ndarray,
    fit_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    lengths = np.asarray(xyz_lengths, dtype=np.float64)
    mask = np.asarray(fit_mask, dtype=np.bool_)
    if lengths.shape[:1] != mask.shape:
        raise ValueError("XYZ reference mask shape mismatch")
    selected = lengths[mask]
    if selected.shape[0] < 2:
        raise CollapseProvenanceError(
            "insufficient XYZ reference rows"
        )
    center = np.median(selected, axis=0)
    q05 = np.percentile(selected, 5.0, axis=0)
    q95 = np.percentile(selected, 95.0, axis=0)
    if np.any(center <= 0.0):
        raise CollapseProvenanceError(
            "XYZ reference contains non-positive center"
        )
    return {
        "median": center,
        "q05": q05,
        "q95": q95,
    }


def _counts_by(
    values: Sequence[Any],
    selected: np.ndarray,
) -> Dict[str, int]:
    array = np.asarray(values)
    mask = np.asarray(selected, dtype=np.bool_)
    if array.shape != mask.shape:
        raise ValueError("count-by shape mismatch")
    counter = Counter(str(value) for value in array[mask])
    return {
        key: int(counter[key])
        for key in sorted(counter)
    }


def _event_category(
    xy_ratio: float,
    xyz_ratio: float,
    spec: ProvenanceAuditSpec,
) -> str:
    if xy_ratio >= spec.severe_xy_ratio:
        return "not_severe_xy"
    if xyz_ratio >= spec.xyz_normal_ratio:
        return "xy_projection_artifact"
    if xyz_ratio < spec.xyz_collapsed_ratio:
        return "true_xyz_collapse"
    return "partial_xyz_collapse"


def build_event_record(
    *,
    row: int,
    horizon: int,
    segment_index: int,
    raw: Mapping[str, np.ndarray],
    components: Mapping[str, np.ndarray],
    xy_reference: np.ndarray,
    xyz_reference: np.ndarray,
    xy_ratio: np.ndarray,
    xyz_ratio: np.ndarray,
    nominal_xyz_ratio: np.ndarray,
    category: str,
) -> Dict[str, Any]:
    positions = np.asarray(raw["positions_xyz"])
    ids = np.asarray(raw["bead_ids"])
    left_position = positions[row, horizon, segment_index]
    right_position = positions[
        row,
        horizon,
        segment_index + 1,
    ]
    return {
        "train_view_local_row": int(row),
        "window_contract_row_index": int(
            np.asarray(raw["window_contract_row_index"])[row]
        ),
        "raw_pickle_frame_index": int(
            np.asarray(raw["raw_info_index"])[row, horizon]
        ),
        "raw_pickle_frame_formula": (
            f"{int(np.asarray(raw['window_t'])[row])} + 1 + "
            f"{horizon}"
        ),
        "source_file": str(
            np.asarray(raw["source_file"])[row]
        ),
        "source_pickle_sha256": str(
            np.asarray(raw["source_pickle_sha256"])[row]
        ),
        "episode_index": int(
            np.asarray(raw["episode_index"])[row]
        ),
        "window_t": int(np.asarray(raw["window_t"])[row]),
        "future_horizon": int(horizon),
        "segment_index": int(segment_index),
        "condition_name": str(
            np.asarray(raw["condition_name"])[row]
        ),
        "visible_seed": int(
            np.asarray(raw["visible_seed"])[row]
        ),
        "pair_group": str(
            np.asarray(raw["pair_group"])[row]
        ),
        "pair_key": str(np.asarray(raw["pair_key"])[row]),
        "episode_group_key": str(
            np.asarray(raw["episode_group_key"])[row]
        ),
        "left_bead_id": int(ids[row, horizon, segment_index]),
        "right_bead_id": int(
            ids[row, horizon, segment_index + 1]
        ),
        "left_position_xyz": left_position.tolist(),
        "right_position_xyz": right_position.tolist(),
        "delta_xyz": np.asarray(
            components["delta_xyz"]
        )[row, horizon, segment_index].tolist(),
        "xy_length": float(
            np.asarray(components["xy_length"])[
                row, horizon, segment_index
            ]
        ),
        "xyz_length": float(
            np.asarray(components["xyz_length"])[
                row, horizon, segment_index
            ]
        ),
        "absolute_dz": float(
            np.asarray(components["abs_dz"])[
                row, horizon, segment_index
            ]
        ),
        "xy_over_xyz": float(
            np.asarray(components["xy_over_xyz"])[
                row, horizon, segment_index
            ]
        ),
        "absolute_dz_over_xyz": float(
            np.asarray(components["abs_dz_over_xyz"])[
                row, horizon, segment_index
            ]
        ),
        "xy_reference_length": float(
            xy_reference[horizon, segment_index]
        ),
        "xyz_reference_length": float(
            xyz_reference[horizon, segment_index]
        ),
        "nominal_source_center_distance":
            NOMINAL_SEGMENT_LENGTH,
        "xy_reference_ratio": float(
            xy_ratio[row, horizon, segment_index]
        ),
        "xyz_reference_ratio": float(
            xyz_ratio[row, horizon, segment_index]
        ),
        "nominal_xyz_ratio": float(
            nominal_xyz_ratio[row, horizon, segment_index]
        ),
        "category": str(category),
    }


def population_audit(
    *,
    name: str,
    population_mask: np.ndarray,
    raw: Mapping[str, np.ndarray],
    components: Mapping[str, np.ndarray],
    xy_reference: np.ndarray,
    xyz_reference: np.ndarray,
    spec: ProvenanceAuditSpec,
) -> Dict[str, Any]:
    mask = np.asarray(population_mask, dtype=np.bool_)
    xy_length = np.asarray(
        components["xy_length"],
        dtype=np.float64,
    )
    xyz_length = np.asarray(
        components["xyz_length"],
        dtype=np.float64,
    )
    xy_ratio = xy_length / xy_reference[None]
    xyz_ratio = xyz_length / xyz_reference[None]
    nominal_xyz_ratio = xyz_length / NOMINAL_SEGMENT_LENGTH

    selected4 = mask[:, None, None]
    severe = (xy_ratio < spec.severe_xy_ratio) & selected4
    moderate = (xy_ratio < spec.moderate_xy_ratio) & selected4
    mild = (xy_ratio < spec.mild_xy_ratio) & selected4
    projection = severe & (
        xyz_ratio >= spec.xyz_normal_ratio
    )
    true_xyz = severe & (
        xyz_ratio < spec.xyz_collapsed_ratio
    )
    partial = severe & ~(projection | true_xyz)

    severe_indices = np.argwhere(severe)
    sorted_indices = sorted(
        [
            (int(row), int(horizon), int(segment))
            for row, horizon, segment in severe_indices
        ],
        key=lambda index: (
            float(xy_ratio[index]),
            int(np.asarray(raw["window_contract_row_index"])[
                index[0]
            ]),
            index[1],
            index[2],
        ),
    )
    top_records = []
    for row, horizon, segment in sorted_indices[
        : spec.top_event_records
    ]:
        category = _event_category(
            float(xy_ratio[row, horizon, segment]),
            float(xyz_ratio[row, horizon, segment]),
            spec,
        )
        top_records.append(
            build_event_record(
                row=row,
                horizon=horizon,
                segment_index=segment,
                raw=raw,
                components=components,
                xy_reference=xy_reference,
                xyz_reference=xyz_reference,
                xy_ratio=xy_ratio,
                xyz_ratio=xyz_ratio,
                nominal_xyz_ratio=nominal_xyz_ratio,
                category=category,
            )
        )

    severe_count = int(np.sum(severe))
    projection_count = int(np.sum(projection))
    true_xyz_count = int(np.sum(true_xyz))
    partial_count = int(np.sum(partial))
    row_severe = np.any(severe, axis=(1, 2))
    source_file = np.asarray(raw["source_file"]).astype(str)
    condition = np.asarray(raw["condition_name"]).astype(str)
    seed = np.asarray(raw["visible_seed"], dtype=np.int64)

    horizon_counter = Counter(
        int(index[1]) for index in severe_indices
    )
    segment_counter = Counter(
        int(index[2]) for index in severe_indices
    )
    source_counter = Counter(
        str(source_file[int(index[0])])
        for index in severe_indices
    )
    condition_counter = Counter(
        str(condition[int(index[0])])
        for index in severe_indices
    )
    seed_counter = Counter(
        int(seed[int(index[0])])
        for index in severe_indices
    )

    return {
        "name": name,
        "row_count": int(np.sum(mask)),
        "segment_event_count": int(
            np.sum(mask) * FUTURE_STEPS * SEGMENT_COUNT
        ),
        "xy_reference_ratio": _safe_stats(
            xy_ratio[mask]
        ),
        "xyz_reference_ratio": _safe_stats(
            xyz_ratio[mask]
        ),
        "nominal_xyz_ratio": _safe_stats(
            nominal_xyz_ratio[mask]
        ),
        "xy_over_xyz": _safe_stats(
            np.asarray(components["xy_over_xyz"])[mask]
        ),
        "absolute_dz_over_xyz": _safe_stats(
            np.asarray(components["abs_dz_over_xyz"])[mask]
        ),
        "threshold_counts": {
            "xy_ratio_below_0p25": severe_count,
            "xy_ratio_below_0p50": int(np.sum(moderate)),
            "xy_ratio_below_0p75": int(np.sum(mild)),
            "rows_with_severe_xy": int(np.sum(row_severe)),
            "xy_projection_artifact_events": projection_count,
            "true_xyz_collapse_events": true_xyz_count,
            "partial_xyz_collapse_events": partial_count,
        },
        "severe_event_fractions": {
            "projection_artifact": (
                float(projection_count / severe_count)
                if severe_count
                else 0.0
            ),
            "true_xyz_collapse": (
                float(true_xyz_count / severe_count)
                if severe_count
                else 0.0
            ),
            "partial_xyz_collapse": (
                float(partial_count / severe_count)
                if severe_count
                else 0.0
            ),
        },
        "severe_events_by_horizon": {
            str(key): int(horizon_counter[key])
            for key in sorted(horizon_counter)
        },
        "severe_events_by_segment": {
            str(key): int(segment_counter[key])
            for key in sorted(segment_counter)
        },
        "severe_events_by_condition": {
            key: int(condition_counter[key])
            for key in sorted(condition_counter)
        },
        "severe_events_by_visible_seed": {
            str(key): int(seed_counter[key])
            for key in sorted(seed_counter)
        },
        "top_source_files": [
            {
                "source_file": key,
                "event_count": int(value),
            }
            for key, value in source_counter.most_common(
                spec.top_source_records
            )
        ],
        "lowest_severe_events": top_records,
        "xy_ratio_sha256": sha256_array(xy_ratio[mask]),
        "xyz_ratio_sha256": sha256_array(xyz_ratio[mask]),
    }


def paired_counterpart_audit(
    *,
    raw: Mapping[str, np.ndarray],
    components: Mapping[str, np.ndarray],
    xy_reference: np.ndarray,
    xyz_reference: np.ndarray,
    spec: ProvenanceAuditSpec,
) -> Dict[str, Any]:
    pair_key = np.asarray(raw["pair_key"]).astype(str)
    condition = np.asarray(raw["condition_name"]).astype(str)
    xy_ratio = (
        np.asarray(components["xy_length"], dtype=np.float64)
        / xy_reference[None]
    )
    xyz_ratio = (
        np.asarray(components["xyz_length"], dtype=np.float64)
        / xyz_reference[None]
    )
    nominal_xyz_ratio = (
        np.asarray(components["xyz_length"], dtype=np.float64)
        / NOMINAL_SEGMENT_LENGTH
    )

    members: MutableMapping[str, List[int]] = defaultdict(list)
    for index, key in enumerate(pair_key):
        members[key].append(index)
    invalid = {
        key: indices
        for key, indices in members.items()
        if len(indices) != 2
        or len(
            set(condition[np.asarray(indices)].tolist())
        ) != 2
    }
    if invalid:
        raise CollapseProvenanceError(
            f"paired train view is malformed: "
            f"{list(invalid.items())[:5]}"
        )

    counterpart = np.empty(pair_key.shape[0], dtype=np.int64)
    for key in sorted(members):
        left, right = members[key]
        counterpart[left] = right
        counterpart[right] = left

    severe = xy_ratio < spec.severe_xy_ratio
    indices = np.argwhere(severe)
    both_severe = 0
    counterpart_not_severe = 0
    same_category = 0
    free_only = 0
    hidden_only = 0
    records = []

    for row, horizon, segment in indices:
        row = int(row)
        horizon = int(horizon)
        segment = int(segment)
        other = int(counterpart[row])
        other_severe = bool(
            severe[other, horizon, segment]
        )
        if other_severe:
            both_severe += 1
        else:
            counterpart_not_severe += 1
            if condition[row] == "free":
                free_only += 1
            else:
                hidden_only += 1
        left_category = _event_category(
            float(xy_ratio[row, horizon, segment]),
            float(xyz_ratio[row, horizon, segment]),
            spec,
        )
        right_category = _event_category(
            float(xy_ratio[other, horizon, segment]),
            float(xyz_ratio[other, horizon, segment]),
            spec,
        )
        if left_category == right_category:
            same_category += 1
        if len(records) < spec.top_event_records:
            record = build_event_record(
                row=row,
                horizon=horizon,
                segment_index=segment,
                raw=raw,
                components=components,
                xy_reference=xy_reference,
                xyz_reference=xyz_reference,
                xy_ratio=xy_ratio,
                xyz_ratio=xyz_ratio,
                nominal_xyz_ratio=nominal_xyz_ratio,
                category=left_category,
            )
            record["counterpart"] = {
                "train_view_local_row": other,
                "window_contract_row_index": int(
                    np.asarray(
                        raw["window_contract_row_index"]
                    )[other]
                ),
                "condition_name": str(condition[other]),
                "source_file": str(
                    np.asarray(raw["source_file"])[other]
                ),
                "raw_pickle_frame_index": int(
                    np.asarray(raw["raw_info_index"])[
                        other, horizon
                    ]
                ),
                "xy_reference_ratio": float(
                    xy_ratio[other, horizon, segment]
                ),
                "xyz_reference_ratio": float(
                    xyz_ratio[other, horizon, segment]
                ),
                "category": right_category,
                "severe_xy": other_severe,
            }
            records.append(record)

    total = int(indices.shape[0])
    return {
        "pair_key_count": len(members),
        "severe_event_count": total,
        "both_conditions_severe_event_instances":
            both_severe,
        "counterpart_not_severe_event_instances":
            counterpart_not_severe,
        "free_only_event_instances": free_only,
        "hidden_only_event_instances": hidden_only,
        "same_category_event_instances": same_category,
        "both_conditions_severe_fraction": (
            float(both_severe / total) if total else 0.0
        ),
        "counterpart_not_severe_fraction": (
            float(counterpart_not_severe / total)
            if total
            else 0.0
        ),
        "records": records,
        "counterpart_index_sha256":
            sha256_array(counterpart),
    }


def source_episode_timeline(
    *,
    episode: LoadedEpisode,
    segment_index: int,
    driver_info_index: int,
    xyz_reference: np.ndarray,
    xy_reference: np.ndarray,
    neighbor_radius: int,
) -> Dict[str, Any]:
    segments = list(
        range(
            max(0, segment_index - neighbor_radius),
            min(
                SEGMENT_COUNT,
                segment_index + neighbor_radius + 1,
            ),
        )
    )
    frames = []
    for info_index, frame in enumerate(episode.frames):
        delta = (
            frame.positions_xyz[1:]
            - frame.positions_xyz[:-1]
        )
        xyz_length = np.linalg.norm(delta, axis=1)
        xy_length = np.linalg.norm(delta[:, :2], axis=1)
        frames.append(
            {
                "raw_pickle_frame_index": info_index,
                "is_driver_frame": (
                    info_index == int(driver_info_index)
                ),
                "segments": [
                    {
                        "segment_index": int(segment),
                        "left_bead_id": int(
                            frame.bead_ids[segment]
                        ),
                        "right_bead_id": int(
                            frame.bead_ids[segment + 1]
                        ),
                        "xy_length": float(xy_length[segment]),
                        "xyz_length": float(
                            xyz_length[segment]
                        ),
                        "xy_reference_ratio_by_horizon0": float(
                            xy_length[segment]
                            / xy_reference[0, segment]
                        ),
                        "xyz_reference_ratio_by_horizon0": float(
                            xyz_length[segment]
                            / xyz_reference[0, segment]
                        ),
                        "xy_over_xyz": float(
                            xy_length[segment] / xyz_length[segment]
                            if xyz_length[segment] > 0.0
                            else 0.0
                        ),
                        "absolute_dz": float(
                            abs(delta[segment, 2])
                        ),
                    }
                    for segment in segments
                ],
            }
        )
    return {
        "source_file": episode.source_file,
        "source_pickle_sha256": episode.source_sha256,
        "frame_count": len(episode.frames),
        "action_count": episode.action_count,
        "driver_segment_index": int(segment_index),
        "driver_raw_pickle_frame_index":
            int(driver_info_index),
        "neighbor_segments": segments,
        "bead_id_order": episode.frames[0].bead_ids.tolist(),
        "frames": frames,
    }


def verify_stage_d1_driver(
    *,
    immutable: Mapping[str, Any],
    raw: Mapping[str, np.ndarray],
    components: Mapping[str, np.ndarray],
    xy_reference: np.ndarray,
    xyz_reference: np.ndarray,
    spec: ProvenanceAuditSpec,
) -> Dict[str, Any]:
    audit = immutable["staged1_gate_audit"]
    drivers = audit["threshold_driver_attribution"][
        "lower_threshold_drivers"
    ]
    driver = min(
        drivers,
        key=lambda record: float(
            record["relative_length_ratio"]
        ),
    )
    source_file = str(driver["source_file"])
    window_row = int(driver["row_index"])
    window_t = int(driver["window_t"])
    horizon = int(driver["horizon"])
    segment = int(driver["segment_index"])
    seed = int(driver["visible_seed"])
    condition = str(driver["condition_name"])

    source_index = np.asarray(
        raw["window_contract_row_index"],
        dtype=np.int64,
    )
    matches = np.flatnonzero(
        (source_index == window_row)
        & (
            np.asarray(raw["source_file"]).astype(str)
            == source_file
        )
        & (
            np.asarray(raw["window_t"], dtype=np.int64)
            == window_t
        )
        & (
            np.asarray(raw["visible_seed"], dtype=np.int64)
            == seed
        )
        & (
            np.asarray(raw["condition_name"]).astype(str)
            == condition
        )
    )
    if matches.shape != (1,):
        raise CollapseProvenanceError(
            "Stage-D.1 lowest driver does not map to one "
            f"train-view row: {matches.tolist()}"
        )
    row = int(matches[0])
    xy_ratio = (
        np.asarray(components["xy_length"])[
            row, horizon, segment
        ]
        / xy_reference[horizon, segment]
    )
    xyz_ratio = (
        np.asarray(components["xyz_length"])[
            row, horizon, segment
        ]
        / xyz_reference[horizon, segment]
    )
    if abs(
        float(xy_ratio)
        - float(driver["relative_length_ratio"])
    ) > 1.0e-6:
        raise CollapseProvenanceError(
            "Stage-D.1 driver XY ratio is not reproduced"
        )
    category = _event_category(
        float(xy_ratio),
        float(xyz_ratio),
        spec,
    )
    event = build_event_record(
        row=row,
        horizon=horizon,
        segment_index=segment,
        raw=raw,
        components=components,
        xy_reference=xy_reference,
        xyz_reference=xyz_reference,
        xy_ratio=(
            np.asarray(components["xy_length"])
            / xy_reference[None]
        ),
        xyz_ratio=(
            np.asarray(components["xyz_length"])
            / xyz_reference[None]
        ),
        nominal_xyz_ratio=(
            np.asarray(components["xyz_length"])
            / NOMINAL_SEGMENT_LENGTH
        ),
        category=category,
    )
    return {
        "stage_d1_driver_record": driver,
        "reconstructed_event": event,
        "mapping_exact": True,
        "provenance_semantics_correction": {
            "stage_d1_row_index": window_row,
            "stage_d1_row_index_meaning": (
                "full 4256-row window-contract index"
            ),
            "raw_pickle_frame_index": int(
                np.asarray(raw["raw_info_index"])[
                    row, horizon
                ]
            ),
            "raw_pickle_frame_formula":
                f"{window_t} + 1 + {horizon}",
        },
    }


def classify_audit(
    *,
    mapping_audit: Mapping[str, Any],
    reconstruction_audit: Mapping[str, Any],
    calibration_population: Mapping[str, Any],
    driver_verification: Mapping[str, Any],
    spec: ProvenanceAuditSpec,
) -> Dict[str, Any]:
    if not bool(mapping_audit["mapping_exact"]):
        root = (
            "phase314b_r256_staged2_window_contract_"
            "mapping_failed"
        )
        next_path = (
            "REPAIR_WINDOW_TO_SOURCE_PROVENANCE_AND_REBUILD_"
            "STATE_V3_CONTRACT"
        )
        locus = "window_provenance_mapping"
    elif (
        float(
            reconstruction_audit["raw_xy_float32_exact_rate"]
        )
        != 1.0
    ):
        root = (
            "phase314b_r256_staged2_raw_xy_cache_"
            "reconstruction_mismatch"
        )
        next_path = (
            "REPAIR_STATE_V3_CABLE_EXTRACTION_AND_REBUILD_"
            "IMMUTABLE_CACHE"
        )
        locus = "cache_extraction"
    elif not bool(
        reconstruction_audit[
            "bead_id_order_stable_across_target_horizons"
        ]
    ):
        root = (
            "phase314b_r256_staged2_bead_order_"
            "contract_failed"
        )
        next_path = (
            "REPAIR_ORDERED_BEAD_SCHEMA_AND_REBUILD_"
            "STATE_V3_CACHE"
        )
        locus = "bead_order"
    else:
        counts = calibration_population["threshold_counts"]
        fractions = calibration_population[
            "severe_event_fractions"
        ]
        severe = int(counts["xy_ratio_below_0p25"])
        projection = float(fractions["projection_artifact"])
        true_xyz = float(fractions["true_xyz_collapse"])
        driver_category = driver_verification[
            "reconstructed_event"
        ]["category"]

        if severe <= 0:
            root = (
                "phase314b_r256_staged2_stage_d1_"
                "collapse_driver_not_reproduced"
            )
            next_path = (
                "REPAIR_STAGE_D1_RATIO_OR_PROVENANCE_AUDIT"
            )
            locus = "staged1_reproduction"
        elif (
            driver_category == "xy_projection_artifact"
            and projection >= spec.projection_dominance_min
            and true_xyz
            <= spec.true_3d_fraction_max_for_projection
        ):
            root = (
                "phase314b_r256_staged2_xy_projection_"
                "collapse_confirmed"
            )
            next_path = (
                "FREEZE_ONE_SIDED_XY_UPPER_SEGMENT_CONTRACT_"
                "AND_REAUDIT_FROZEN_REVERSE"
            )
            locus = "xy_projection_contract"
        elif true_xyz >= spec.true_3d_fraction_min:
            root = (
                "phase314b_r256_staged2_true_xyz_segment_"
                "collapse_present"
            )
            next_path = (
                "AUDIT_DEFORMABLERAVENS_CABLE_CONSTRAINTS_"
                "FOR_TRUE_3D_COLLAPSE"
            )
            locus = "source_xyz_collapse"
        else:
            root = (
                "phase314b_r256_staged2_mixed_projection_"
                "and_xyz_collapse"
            )
            next_path = (
                "STRATIFY_XY_PROJECTION_AND_TRUE_XYZ_COLLAPSE_"
                "BEFORE_GATE_FREEZE"
            )
            locus = "mixed_source_geometry"

    return {
        "root_cause": root,
        "required_next_path": next_path,
        "primary_failure_locus": locus,
        "cache_mapping_exact": bool(
            mapping_audit["mapping_exact"]
        ),
        "raw_xy_cache_exact": (
            float(
                reconstruction_audit[
                    "raw_xy_float32_exact_rate"
                ]
            )
            == 1.0
        ),
        "bead_order_stable": bool(
            reconstruction_audit[
                "bead_id_order_stable_across_target_horizons"
            ]
        ),
        "driver_category": driver_verification[
            "reconstructed_event"
        ]["category"],
        "calibration_projection_artifact_fraction":
            calibration_population["severe_event_fractions"][
                "projection_artifact"
            ],
        "calibration_true_xyz_collapse_fraction":
            calibration_population["severe_event_fractions"][
                "true_xyz_collapse"
            ],
        "calibration_partial_xyz_collapse_fraction":
            calibration_population["severe_event_fractions"][
                "partial_xyz_collapse"
            ],
    }


def run_audit(
    *,
    root: Path,
    spec: Optional[ProvenanceAuditSpec] = None,
) -> Dict[str, Any]:
    active_spec = (
        ProvenanceAuditSpec() if spec is None else spec
    )
    active_spec.validate()
    repository_root = Path(root).resolve()

    immutable = validate_immutable_inputs(repository_root)
    source_audit = source_logic_audit(repository_root)
    if not source_audit["all_confirmed"]:
        raise CollapseProvenanceError(
            "source ordering/projection assumptions changed"
        )

    train_view = stageb.load_npz_strict(
        repository_root / stageb.STAGEA_TRAIN_VIEW,
        required_keys=TRAIN_REQUIRED_KEYS,
    )
    full_contract = stageb.load_npz_strict(
        repository_root / stageb.STAGEA_CONTRACT,
        required_keys=CONTRACT_REQUIRED_KEYS,
    )
    mapping = validate_contract_mapping(
        train_view,
        full_contract,
    )
    raw, reconstruction, store = reconstruct_raw_targets(
        repository_root=repository_root,
        train_view=train_view,
        full_contract=full_contract,
        spec=active_spec,
    )

    groups = np.asarray(
        train_view["episode_group_key"]
    ).astype(str)
    stageb_train_mask, probe_mask, stageb_mapping = (
        stageb.deterministic_group_split(
            groups,
            folds=stageb.DiagnosticSpec().group_folds,
            probe_fold=stageb.DiagnosticSpec().probe_fold,
        )
    )
    fit_mask, calibration_mask, gate_split = (
        staged.deterministic_gate_split(
            groups,
            stageb_train_mask,
            spec=staged.RecalibrationSpec(),
        )
    )

    gate, _ = staged1.load_stage_d_gate(
        repository_root / STAGE_D_GATE
    )
    if gate_split["fit_group_sha256"] != gate.fit_group_sha256:
        raise CollapseProvenanceError(
            "Stage-D fit-group identity changed"
        )
    if (
        gate_split["calibration_group_sha256"]
        != gate.calibration_group_sha256
    ):
        raise CollapseProvenanceError(
            "Stage-D calibration-group identity changed"
        )

    components = segment_components(
        raw["positions_xyz"]
    )
    xyz_reference_record = fit_xyz_reference(
        components["xyz_length"],
        fit_mask,
    )
    xyz_reference = xyz_reference_record["median"]
    xy_reference = np.exp(
        gate.reference.center_log.astype(np.float64)
    )

    populations = {
        "reference_fit": fit_mask,
        "gate_calibration": calibration_mask,
        "combined_stageb_training": stageb_train_mask,
        "frozen_probe": probe_mask,
        "all_train_full_horizon": np.ones(
            EXPECTED_TRAIN_FULL_ROWS,
            dtype=np.bool_,
        ),
    }
    population_results = {
        name: population_audit(
            name=name,
            population_mask=mask,
            raw=raw,
            components=components,
            xy_reference=xy_reference,
            xyz_reference=xyz_reference,
            spec=active_spec,
        )
        for name, mask in populations.items()
    }

    driver = verify_stage_d1_driver(
        immutable=immutable,
        raw=raw,
        components=components,
        xy_reference=xy_reference,
        xyz_reference=xyz_reference,
        spec=active_spec,
    )
    driver_event = driver["reconstructed_event"]
    driver_episode = store.get(
        driver_event["source_file"],
        driver_event["source_pickle_sha256"],
    )
    timeline = source_episode_timeline(
        episode=driver_episode,
        segment_index=int(driver_event["segment_index"]),
        driver_info_index=int(
            driver_event["raw_pickle_frame_index"]
        ),
        xyz_reference=xyz_reference,
        xy_reference=xy_reference,
        neighbor_radius=active_spec.timeline_neighbor_radius,
    )

    paired = paired_counterpart_audit(
        raw=raw,
        components=components,
        xy_reference=xy_reference,
        xyz_reference=xyz_reference,
        spec=active_spec,
    )
    classification = classify_audit(
        mapping_audit=mapping,
        reconstruction_audit=reconstruction,
        calibration_population=population_results[
            "gate_calibration"
        ],
        driver_verification=driver,
        spec=active_spec,
    )

    references = {
        "stage_d_xy_reference_length": {
            "shape": list(xy_reference.shape),
            "sha256": sha256_array(xy_reference),
            "stats": _safe_stats(xy_reference),
        },
        "source_xyz_reference_length": {
            "fit_population": "Stage-D reference-fit groups only",
            "median_shape": list(xyz_reference.shape),
            "median_sha256": sha256_array(xyz_reference),
            "median_stats": _safe_stats(xyz_reference),
            "q05_sha256": sha256_array(
                xyz_reference_record["q05"]
            ),
            "q95_sha256": sha256_array(
                xyz_reference_record["q95"]
            ),
        },
        "deformable_ravens_nominal_center_distance": {
            "radius": NOMINAL_CABLE_RADIUS,
            "formula": "2 * radius * sqrt(2)",
            "value": NOMINAL_SEGMENT_LENGTH,
        },
    }

    result = {
        "phase": PHASE,
        "phase_id": PHASE_ID,
        "schema": "phase314b_r256_staged2_worker_result_v1",
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": classification["root_cause"],
        "required_next_path":
            classification["required_next_path"],
        "audit_spec": asdict(active_spec),
        "immutable_inputs": {
            "base_commit": immutable["base_commit"],
            "base_file_sha256":
                immutable["base_file_sha256"],
        },
        "source_logic_audit": source_audit,
        "contract_mapping_audit": mapping,
        "raw_reconstruction_audit": reconstruction,
        "split": {
            "total_rows": EXPECTED_TRAIN_FULL_ROWS,
            "stageb_training_rows": int(
                np.sum(stageb_train_mask)
            ),
            "stageb_probe_rows": int(np.sum(probe_mask)),
            "reference_fit_rows": int(np.sum(fit_mask)),
            "gate_calibration_rows": int(
                np.sum(calibration_mask)
            ),
            "stageb_group_count": len(stageb_mapping),
            "reference_fit_groups":
                gate_split["fit_group_count"],
            "gate_calibration_groups":
                gate_split["calibration_group_count"],
            "group_boundaries_intact": True,
        },
        "geometry_references": references,
        "population_geometry_audit": population_results,
        "stage_d1_driver_verification": driver,
        "stage_d1_driver_source_timeline": timeline,
        "paired_counterpart_audit": paired,
        "classification": classification,
        "provenance_row_semantics_corrected": True,
        "source_pickle_loaded_read_only": True,
        "raw_xyz_used_for_audit_only": True,
        "deformable_ravens_executed": False,
        "deformable_ravens_modified": False,
        "gate_selected": False,
        "threshold_changed": False,
        "model_or_branch_repaired": False,
        "diffusion_training": False,
        "reverse_sampling": False,
        "formal_pilot_run": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "candidate_tensor_persisted": False,
        "npz_saved": False,
        "formal_idm_training": False,
        "action_diverse_data_collection": False,
        "candidate_execution": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }
    return result


def identity_projection(
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    return {
        "root_cause": result["root_cause"],
        "required_next_path": result["required_next_path"],
        "source_logic_audit": result["source_logic_audit"],
        "contract_mapping_audit":
            result["contract_mapping_audit"],
        "raw_reconstruction_audit":
            result["raw_reconstruction_audit"],
        "split": result["split"],
        "geometry_references": result["geometry_references"],
        "population_geometry_audit":
            result["population_geometry_audit"],
        "stage_d1_driver_verification":
            result["stage_d1_driver_verification"],
        "stage_d1_driver_source_timeline":
            result["stage_d1_driver_source_timeline"],
        "paired_counterpart_audit":
            result["paired_counterpart_audit"],
        "classification": result["classification"],
    }


def compare_worker_results(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
) -> Dict[str, Any]:
    left_payload = stable_json_bytes(identity_projection(left))
    right_payload = stable_json_bytes(identity_projection(right))
    return {
        "exact": left_payload == right_payload,
        "left_sha256": sha256_bytes(left_payload),
        "right_sha256": sha256_bytes(right_payload),
        "mapping_exact": (
            left["contract_mapping_audit"]
            == right["contract_mapping_audit"]
        ),
        "raw_reconstruction_exact": (
            left["raw_reconstruction_audit"]
            == right["raw_reconstruction_audit"]
        ),
        "population_audit_exact": (
            left["population_geometry_audit"]
            == right["population_geometry_audit"]
        ),
        "classification_exact": (
            left["classification"] == right["classification"]
        ),
    }

"""Resume5 same-device residual determinism and functional reproduction audit.

This module is additive audit code.  It deliberately instruments the existing
r2.5.1 training function at its module boundaries instead of modifying or
copying that function.  Tensor values are hashed in memory; weights,
predictions, RNG states, and optimizer states are never serialized to disk.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ccda_phase3.phase314b_r254_resume3_prediction_adapter import (
    stochastic_state_fingerprint,
)


PHASE = "phase3_14b_r254_resume5"
SCHEMA = "phase314b_r254_resume5_same_device_residual_determinism_v1"
BASE_EVIDENCE_COMMIT = "4388d8536fd619f36f13b825d67d647ce5fd718a"
EXPECTED_SUBMODULE_COMMIT = "633a88752445cf5d6776ed374fdbbdb35f93050c"
EXPECTED_CACHE_SHA256 = (
    "3cc512f650557c81c3b81f4f128a5b55360b77d3f664fdd6607666936285fbe8"
)
EXPECTED_CONTRACT_SHA256 = (
    "fa2725ca40da2499008360f13d291d2ce8694e6393910b0522fe800f4f37cdcc"
)
EXPECTED_RESUME4_TEST_GATE_SHA256 = (
    "39df72981d552573be7bfc4d5a0dc794fd2b244a909ace3a65cdef3bfe68f988"
)
EXPECTED_RESUME4_PREFLIGHT_SHA256 = (
    "a7202ed03a508a96195ce871c263cd7758f706f05882e5b32ab8457fd7f4a6ac"
)
EXPECTED_RESUME4_AUDIT_SHA256 = (
    "07b1d81e82cab82c1901883808519f7d1468a6fcdb8add7c87001362aa8bf312"
)
EXPECTED_RESUME4_SUMMARY_SHA256 = (
    "9772306891221208afb8e44ef54aea7098379a38cdddedb735d22b5c19af4667"
)
EXPECTED_RESUME4_MARKDOWN_SHA256 = (
    "04e5432a6b82099b85106e8ae9a3d3abab7b5ca7989d9c92caab09f48f37d124"
)
EXPECTED_RESUME3_PILOT_SHA256 = (
    "d28c4b40c16ba852a2dc3cc42cfee548defeb6aa7ed3717203690282b984f36e"
)
EXPECTED_CURRENT_PRIOR_STATE_SHA256 = (
    "8610337d4e869764c7315280056f6af9151ccf2d9872fc47d3afc9dca066e904"
)
EXPECTED_CURRENT_PRIOR_PREDICTION_SHA256 = (
    "70c3a6bde584ee6973190823e37548b69121ddf7fbc4c0cd5fe069a4d69ae7b9"
)
EXPECTED_BASE_TEST_COUNT = 425
EXPECTED_RESUME5_TEST_COUNT = 13
EXPECTED_SCOPED_TEST_COUNT = EXPECTED_BASE_TEST_COUNT + EXPECTED_RESUME5_TEST_COUNT
EXPECTED_REPEAT_COUNT = 3

RESUME4_TEST_GATE_PATH = "reports/phase3_14b_r254_resume4_test_gate_summary.json"
RESUME4_PREFLIGHT_PATH = "reports/phase3_14b_r254_resume4_preflight_summary.json"
RESUME4_AUDIT_PATH = "reports/phase3_14b_r254_resume4_reproduction_audit_summary.json"
RESUME4_SUMMARY_PATH = "reports/phase3_14b_r254_resume4_summary.json"
RESUME4_MARKDOWN_PATH = "reports/phase3_14b_r254_resume4_report.md"
RESUME3_PILOT_PATH = "reports/phase3_14b_r254_resume3_pilot_summary.json"
RESUME5_TEST_PATH = "tests/test_phase314b_resume5_residual_determinism.py"

DIAGNOSTIC_OBJECTIVE_NAMES = (
    "v_only_frozen_control",
    "ordered_mean_raw_g100",
    "ordered_cvar_contract_g010",
)
CONTRACT_KEYS = (
    "historical_exact_reconstruction_pass",
    "branch_transport_pass",
    "one_step_physical_pass",
    "ordered_topology_pass",
    "final_horizon_cable_geometry_pass",
    "all_earlier_horizon_cable_geometry_pass",
)
FUNCTIONAL_CABLE_KEYS = (
    "branch_transport_pass",
    "ordered_topology_pass",
    "final_horizon_cable_geometry_pass",
    "all_earlier_horizon_cable_geometry_pass",
)
NUMERIC_KEYS = (
    "full_z_mse",
    "cable_z_mse",
    "robot_proxy_z_mse",
    "cable_error_fraction",
    "robot_proxy_error_fraction",
)
IDENTITY_FIELDS = (
    "residual_initial_state_sha256",
    "residual_final_state_sha256",
    "optimizer_initial_state_sha256",
    "optimizer_final_state_sha256",
    "prediction_sha256",
    "training_loss_history_sha256",
    "gradient_history_sha256",
    "source_exposure_counts_sha256",
    "training_rng_fingerprint",
)

SOURCE_PATHS = (
    "ccda_phase3/phase314b_r254_resume5_residual_determinism.py",
    "scripts/phase3_14b_r254_resume5_worker.py",
    "scripts/phase3_14b_r254_resume5_test_gate.py",
    "scripts/phase3_14b_r254_resume5_preflight.py",
    "scripts/phase3_14b_r254_resume5_run_audit.py",
    "scripts/phase3_14b_r254_resume5_finalize.py",
    "scripts/phase3_14b_r254_resume5_blocked.py",
    "scripts/phase3_14b_r254_resume5_run.sh",
    RESUME5_TEST_PATH,
)


@dataclass(frozen=True)
class Resume5Spec:
    repeat_count: int = EXPECTED_REPEAT_COUNT
    expected_gpu_name: str = "NVIDIA GeForce RTX 4090"
    prior_steps: int = 5000
    residual_steps: int = 8000
    batch_size: int = 64
    prior_learning_rate: float = 1.0e-3
    residual_learning_rate: float = 1.0e-3
    calibration_batches: int = 8
    common_prior_seed: int = 102000
    common_training_seed: int = 102000

    def validate(self) -> None:
        for name in (
            "repeat_count",
            "prior_steps",
            "residual_steps",
            "batch_size",
            "calibration_batches",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.repeat_count != EXPECTED_REPEAT_COUNT:
            raise ValueError("Resume5 requires exactly three isolated repeats")
        for name in ("prior_learning_rate", "residual_learning_rate"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not self.expected_gpu_name:
            raise ValueError("expected_gpu_name is empty")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_sha256(root: Path) -> Dict[str, str]:
    base = Path(root)
    return {path: sha256_file(base / path) for path in SOURCE_PATHS}


def load_json(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    assert_json_summary_safe(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_text_once(path: Path, value: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(str(value))


def git_output(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=Path(root), text=True, stderr=subprocess.STDOUT
    ).strip()


def assert_only_allowed_worktree_paths(root: Path, allowed: Sequence[str]) -> None:
    allowed_set = {Path(value).as_posix() for value in allowed}
    output = git_output(root, "status", "--porcelain", "--untracked-files=all")
    observed = set()
    for line in output.splitlines():
        if not line:
            continue
        value = line[3:]
        if " -> " in value:
            value = value.split(" -> ", 1)[1]
        observed.add(Path(value).as_posix())
    unexpected = sorted(observed - allowed_set)
    if unexpected:
        raise RuntimeError("unexpected worktree paths: " + ", ".join(unexpected))


def _canonical_update(digest: Any, value: Any) -> None:
    """Hash nested Python/NumPy/Torch values with explicit type boundaries."""
    if value is None:
        digest.update(b"N;")
    elif isinstance(value, bool):
        digest.update(b"B1;" if value else b"B0;")
    elif isinstance(value, (int, np.integer)) and not isinstance(value, bool):
        digest.update(b"I" + str(int(value)).encode("ascii") + b";")
    elif isinstance(value, (float, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("non-finite float cannot be canonically hashed")
        digest.update(b"F" + struct.pack(">d", number) + b";")
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + str(len(encoded)).encode("ascii") + b":" + encoded + b";")
    elif isinstance(value, bytes):
        digest.update(b"Y" + str(len(value)).encode("ascii") + b":" + value + b";")
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"T")
        _canonical_update(digest, str(tensor.dtype))
        _canonical_update(digest, list(tensor.shape))
        payload = tensor.numpy().tobytes(order="C")
        _canonical_update(digest, payload)
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("object arrays cannot be canonically hashed")
        digest.update(b"A")
        _canonical_update(digest, str(array.dtype))
        _canonical_update(digest, list(array.shape))
        _canonical_update(digest, array.tobytes(order="C"))
    elif isinstance(value, Mapping):
        digest.update(b"M" + str(len(value)).encode("ascii") + b"{")
        ordered = sorted(value.items(), key=lambda item: (type(item[0]).__name__, repr(item[0])))
        for key, item in ordered:
            if not isinstance(key, (str, int, np.integer)):
                raise TypeError(f"unsupported mapping key type: {type(key).__name__}")
            _canonical_update(digest, int(key) if isinstance(key, (int, np.integer)) else key)
            _canonical_update(digest, item)
        digest.update(b"};")
    elif isinstance(value, (list, tuple)):
        digest.update(b"L" + str(len(value)).encode("ascii") + b"[")
        for item in value:
            _canonical_update(digest, item)
        digest.update(b"];")
    else:
        raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_sha256(value: Any) -> str:
    digest = hashlib.sha256()
    _canonical_update(digest, value)
    return digest.hexdigest()


def tensor_fingerprint(value: torch.Tensor) -> Dict[str, Any]:
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    return {
        "sha256": canonical_sha256(tensor),
        "shape": [int(item) for item in tensor.shape],
        "dtype": str(tensor.dtype),
    }


def module_state_sha256(module: torch.nn.Module) -> str:
    return canonical_sha256(dict(module.state_dict()))


def optimizer_state_sha256(optimizer: torch.optim.Optimizer) -> str:
    return canonical_sha256(optimizer.state_dict())


def assert_json_summary_safe(value: Any, path: str = "root") -> None:
    """Reject tensor/array payloads and non-finite values before evidence writes."""
    if isinstance(value, (torch.Tensor, np.ndarray)):
        raise TypeError(f"runtime tensor/array payload forbidden at {path}")
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite evidence value at {path}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string evidence key at {path}")
            assert_json_summary_safe(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_json_summary_safe(item, f"{path}[{index}]")
        return
    raise TypeError(f"unsupported evidence value at {path}: {type(value).__name__}")


class _SamplerTrace:
    def __init__(self, tag: str, sampler: Any) -> None:
        self.tag = str(tag)
        self.sampler = sampler
        self.calls = 0
        self.digest = hashlib.sha256()

    def observe(self, index: torch.Tensor) -> None:
        self.calls += 1
        _canonical_update(self.digest, int(self.calls))
        _canonical_update(self.digest, index)

    def report(self) -> Dict[str, Any]:
        counts = self.sampler.count_report()
        return {
            "tag": self.tag,
            "call_count": int(self.calls),
            "index_sequence_sha256": self.digest.hexdigest(),
            "count_report": counts,
        }


class ResidualTrainingRecorder:
    """Temporary boundary instrumentation for one unchanged training call."""

    def __init__(self, module: Any) -> None:
        self.module = module
        self.model: Optional[torch.nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.residual_initial_state_sha256: Optional[str] = None
        self.optimizer_initial_state_sha256: Optional[str] = None
        self.global_rng_after_model_init: Optional[str] = None
        self.samplers: Dict[str, _SamplerTrace] = {}
        self.generator_refs: Dict[str, torch.Generator] = {}
        self.generator_digests: Dict[str, Any] = {}
        self.generator_calls: Dict[str, int] = {}
        self._originals: Dict[str, Any] = {}

    def _tag(self) -> str:
        return "training" if self.optimizer is not None else "calibration"

    def install(self) -> None:
        if self._originals:
            raise RuntimeError("recorder is already installed")
        module = self.module
        self._originals = {
            "instantiate_snapshot_model": module.instantiate_snapshot_model,
            "_optimizer": module._optimizer,
            "SourceBatchSampler": module.SourceBatchSampler,
            "_sample_timesteps": module._sample_timesteps,
        }
        recorder = self

        def instantiate_wrapper(*args: Any, **kwargs: Any) -> Any:
            model = recorder._originals["instantiate_snapshot_model"](*args, **kwargs)
            recorder.model = model
            recorder.residual_initial_state_sha256 = module_state_sha256(model.residual_v)
            recorder.global_rng_after_model_init = stochastic_state_fingerprint()
            return model

        def optimizer_wrapper(*args: Any, **kwargs: Any) -> Any:
            optimizer = recorder._originals["_optimizer"](*args, **kwargs)
            recorder.optimizer = optimizer
            recorder.optimizer_initial_state_sha256 = optimizer_state_sha256(optimizer)
            return optimizer

        base_sampler = self._originals["SourceBatchSampler"]

        class RecordingSourceBatchSampler(base_sampler):  # type: ignore[misc,valid-type]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                tag = recorder._tag()
                if tag in recorder.samplers:
                    raise RuntimeError(f"duplicate {tag} sampler")
                recorder.samplers[tag] = _SamplerTrace(tag, self)

            def next_indices(self) -> torch.Tensor:
                index = super().next_indices()
                recorder.samplers[recorder._tag()].observe(index)
                return index

        def sample_timesteps_wrapper(*args: Any, **kwargs: Any) -> torch.Tensor:
            generator = kwargs.get("generator")
            if generator is None:
                raise RuntimeError("instrumented timestep call lacks generator")
            tag = recorder._tag()
            digest = recorder.generator_digests.setdefault(tag, hashlib.sha256())
            recorder.generator_refs[tag] = generator
            recorder.generator_calls[tag] = recorder.generator_calls.get(tag, 0) + 1
            _canonical_update(digest, ("before", generator.get_state()))
            result = recorder._originals["_sample_timesteps"](*args, **kwargs)
            _canonical_update(digest, ("timesteps", result))
            _canonical_update(digest, ("after_timestep", generator.get_state()))
            return result

        module.instantiate_snapshot_model = instantiate_wrapper
        module._optimizer = optimizer_wrapper
        module.SourceBatchSampler = RecordingSourceBatchSampler
        module._sample_timesteps = sample_timesteps_wrapper

    def uninstall(self) -> None:
        for name, value in self._originals.items():
            setattr(self.module, name, value)
        self._originals = {}

    def finish(self, result: Mapping[str, Any]) -> Dict[str, Any]:
        if self.model is None or self.optimizer is None:
            raise RuntimeError("training did not construct model and optimizer")
        prediction = result.get("_true_prediction_z")
        if not isinstance(prediction, torch.Tensor):
            raise RuntimeError("training did not return the fixed-bank prediction")
        streams: Dict[str, Any] = {}
        for tag, digest in sorted(self.generator_digests.items()):
            generator = self.generator_refs[tag]
            _canonical_update(digest, ("final_after_noise", generator.get_state()))
            streams[tag] = {
                "call_count": int(self.generator_calls[tag]),
                "state_sequence_sha256": digest.hexdigest(),
                "final_state_sha256": canonical_sha256(generator.get_state()),
            }
        sampler_reports = {
            tag: trace.report() for tag, trace in sorted(self.samplers.items())
        }
        history = result.get("residual_history", [])
        gradients = result.get("gradient_audit", [])
        exposure = result.get("source_exposure", {}).get("counts", [])
        training_rng = {
            "global_after_model_init": self.global_rng_after_model_init,
            "global_after_training_and_evaluation": stochastic_state_fingerprint(),
            "local_generators": streams,
            "samplers": sampler_reports,
        }
        identity = {
            "residual_initial_state_sha256": self.residual_initial_state_sha256,
            "residual_final_state_sha256": module_state_sha256(self.model.residual_v),
            "optimizer_initial_state_sha256": self.optimizer_initial_state_sha256,
            "optimizer_final_state_sha256": optimizer_state_sha256(self.optimizer),
            "prediction_sha256": tensor_fingerprint(prediction)["sha256"],
            "prediction_fingerprint": tensor_fingerprint(prediction),
            "training_loss_history": history,
            "training_loss_history_sha256": canonical_sha256(history),
            "gradient_history": gradients,
            "gradient_history_sha256": canonical_sha256(gradients),
            "source_exposure_counts": exposure,
            "source_exposure_counts_sha256": canonical_sha256(exposure),
            "training_rng": training_rng,
            "training_rng_fingerprint": canonical_sha256(training_rng),
        }
        missing = [name for name in IDENTITY_FIELDS if not identity.get(name)]
        if missing:
            raise RuntimeError(f"residual identity fields are incomplete: {missing}")
        assert_json_summary_safe(identity)
        return identity


@contextmanager
def instrument_residual_training(module: Any) -> Iterator[ResidualTrainingRecorder]:
    recorder = ResidualTrainingRecorder(module)
    recorder.install()
    try:
        yield recorder
    finally:
        recorder.uninstall()


def parse_worker_stdout(output: str) -> Dict[str, Any]:
    prefix = "RESUME5_RESULT_JSON="
    matches = [line[len(prefix):] for line in str(output).splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError("worker must emit exactly one Resume5 JSON sentinel")
    value = json.loads(matches[0])
    if not isinstance(value, dict):
        raise ValueError("worker sentinel payload must be an object")
    assert_json_summary_safe(value)
    return value


def _comparison_rows(values: Sequence[Any]) -> Dict[str, Any]:
    hashes = [canonical_sha256(value) for value in values]
    return {
        "repeat_sha256": hashes,
        "exact": bool(hashes and len(set(hashes)) == 1),
    }


def compare_repeat_matrix(repeats: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    if len(repeats) != EXPECTED_REPEAT_COUNT:
        raise ValueError("expected exactly three worker repeats")
    indices = [int(item.get("repeat_index", -1)) for item in repeats]
    if indices != list(range(EXPECTED_REPEAT_COUNT)):
        raise ValueError(f"repeat indices changed: {indices}")
    runtime = _comparison_rows([item.get("runtime") for item in repeats])
    prior = _comparison_rows([item.get("prior") for item in repeats])
    models: Dict[str, Any] = {}
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        values = []
        fields: Dict[str, Any] = {}
        for repeat in repeats:
            variants = repeat.get("variants")
            if not isinstance(variants, Mapping) or name not in variants:
                raise ValueError(f"worker repeat missing variant {name}")
            values.append(variants[name])
        for field in IDENTITY_FIELDS:
            field_values = [item.get("identity", {}).get(field) for item in values]
            fields[field] = _comparison_rows(field_values)
        complete = _comparison_rows(values)
        models[name] = {
            "identity_fields": fields,
            "complete_variant": complete,
            "exact": bool(
                complete["exact"]
                and all(item["exact"] for item in fields.values())
            ),
        }
    return {
        "runtime": runtime,
        "prior": prior,
        "models": models,
        "same_device_exact": bool(
            runtime["exact"]
            and prior["exact"]
            and all(item["exact"] for item in models.values())
        ),
    }


def _resume4_expected_observed(
    resume4_audit: Mapping[str, Any], name: str
) -> Tuple[Dict[str, bool], Dict[str, float], Dict[str, bool], Dict[str, float]]:
    models = resume4_audit.get("models")
    if not isinstance(models, Mapping):
        raise ValueError("Resume4 model audit is missing")
    model = models.get(name)
    if not isinstance(model, Mapping):
        raise ValueError(f"Resume4 model audit missing {name}")
    historical_contract: Dict[str, bool] = {}
    current_contract: Dict[str, bool] = {}
    historical_numeric: Dict[str, float] = {}
    current_numeric: Dict[str, float] = {}
    checks = model.get("checks")
    if not isinstance(checks, list):
        raise ValueError("Resume4 checks must be a list")
    for item in checks:
        if not isinstance(item, Mapping):
            raise ValueError("Resume4 check must be an object")
        check = str(item.get("check"))
        if check.startswith("contract."):
            key = check.split(".", 1)[1]
            historical_contract[key] = bool(item.get("expected"))
            current_contract[key] = bool(item.get("observed"))
        elif check.startswith("state_group_z_metrics."):
            key = check.split(".", 1)[1]
            historical_numeric[key] = float(item.get("expected"))
            current_numeric[key] = float(item.get("observed"))
    if set(historical_contract) != set(CONTRACT_KEYS):
        raise ValueError(f"Resume4 contract matrix changed for {name}")
    if set(historical_numeric) != set(NUMERIC_KEYS):
        raise ValueError(f"Resume4 numeric matrix changed for {name}")
    return historical_contract, historical_numeric, current_contract, current_numeric


def compare_fresh_to_resume3(
    fresh_variant: Mapping[str, Any],
    expected_contract: Mapping[str, bool],
    expected_numeric: Mapping[str, float],
) -> Dict[str, Any]:
    observed_contract = fresh_variant.get("contracts")
    observed_numeric = fresh_variant.get("state_group_z_metrics")
    if not isinstance(observed_contract, Mapping) or not isinstance(observed_numeric, Mapping):
        raise ValueError("fresh variant lacks separation contracts/metrics")
    rows = []
    for key in CONTRACT_KEYS:
        expected = bool(expected_contract[key])
        observed = bool(observed_contract.get(key))
        rows.append({
            "path": f"contract.{key}",
            "kind": "boolean",
            "expected": expected,
            "observed": observed,
            "exact": observed == expected,
        })
    for key in NUMERIC_KEYS:
        expected = float(expected_numeric[key])
        observed = float(observed_numeric.get(key))
        rows.append({
            "path": f"state_group_z_metrics.{key}",
            "kind": "float64_scalar",
            "expected": expected,
            "observed": observed,
            "absolute_difference": abs(observed - expected),
            "exact": canonical_sha256(observed) == canonical_sha256(expected),
        })
    return {
        "checks": rows,
        "exact": bool(all(item["exact"] for item in rows)),
    }


def functional_reproduction_contract(
    historical_contract: Mapping[str, bool],
    current_contract: Mapping[str, bool],
) -> Dict[str, Any]:
    """Predeclared cable-only, zero-new-tolerance non-regression contract."""
    rows = []
    for key in FUNCTIONAL_CABLE_KEYS:
        historical = bool(historical_contract[key])
        current = bool(current_contract[key])
        rows.append({
            "contract": key,
            "historical": historical,
            "current": current,
            "exact_agreement": historical == current,
            "historical_true_regressed": bool(historical and not current),
            "current_required_gate_pass": current,
            "pass": bool(current and not (historical and not current)),
        })
    return {
        "schema": "phase314b_r254_resume5_cable_functional_nonregression_v1",
        "rule": "all fixed cable gates pass now and no historical true gate becomes false",
        "new_numeric_tolerance_introduced": False,
        "excluded_from_gate": {
            "historical_exact_reconstruction_pass": (
                "full-state gate is robot-proxy-confounded"
            ),
            "one_step_physical_pass": (
                "mixed gate includes robot-proxy end-effector quaternion validity"
            ),
            "full_z_mse": "contains robot-proxy dimensions",
            "robot_proxy_z_mse": "pinned logger schema has confirmed defects",
            "robot_proxy_error_fraction": "pinned logger schema has confirmed defects",
            "cable_z_mse": "reported diagnostically; no outcome-fitted tolerance authorized",
            "cable_error_fraction": "reported diagnostically; no outcome-fitted tolerance authorized",
        },
        "checks": rows,
        "pass": bool(all(item["pass"] for item in rows)),
    }


def build_resume5_audit(
    *,
    repeats: Sequence[Mapping[str, Any]],
    resume4_audit: Mapping[str, Any],
    spec: Resume5Spec = Resume5Spec(),
) -> Dict[str, Any]:
    spec.validate()
    repeat_comparison = compare_repeat_matrix(repeats)
    first = repeats[0]
    prior = first.get("prior")
    if not isinstance(prior, Mapping):
        raise ValueError("worker prior summary is missing")
    prior_contract = bool(
        prior.get("state_sha256") == EXPECTED_CURRENT_PRIOR_STATE_SHA256
        and prior.get("prediction_sha256") == EXPECTED_CURRENT_PRIOR_PREDICTION_SHA256
        and repeat_comparison["prior"]["exact"]
    )
    model_reports: Dict[str, Any] = {}
    resume3_exact = True
    functional_pass = True
    strict_boolean_exact = True
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        historical_c, historical_n, resume3_c, resume3_n = _resume4_expected_observed(
            resume4_audit, name
        )
        fresh = first["variants"][name]
        current = compare_fresh_to_resume3(fresh, resume3_c, resume3_n)
        functional = functional_reproduction_contract(
            historical_c, fresh["contracts"]
        )
        strict_rows = [
            {
                "contract": key,
                "historical": historical_c[key],
                "current": bool(fresh["contracts"][key]),
                "exact": historical_c[key] == bool(fresh["contracts"][key]),
            }
            for key in CONTRACT_KEYS
        ]
        strict = all(item["exact"] for item in strict_rows)
        model_reports[name] = {
            "same_device_repeat_exact": repeat_comparison["models"][name],
            "fresh_exact_match_to_resume3": current,
            "historical_boolean_exact_agreement": {
                "checks": strict_rows,
                "pass": strict,
            },
            "functional_cable_nonregression": functional,
            "historical_numeric_differences": {
                key: {
                    "historical": historical_n[key],
                    "current": float(fresh["state_group_z_metrics"][key]),
                    "absolute_difference": abs(
                        float(fresh["state_group_z_metrics"][key]) - historical_n[key]
                    ),
                    "gate": False,
                }
                for key in NUMERIC_KEYS
            },
        }
        resume3_exact = bool(resume3_exact and current["exact"])
        functional_pass = bool(functional_pass and functional["pass"])
        strict_boolean_exact = bool(strict_boolean_exact and strict)

    same_device = bool(repeat_comparison["same_device_exact"])
    functional_reproduction_pass = bool(
        prior_contract and same_device and resume3_exact and functional_pass
    )
    if not prior_contract:
        root = "phase314b_r254_resume5_functional_prior_reproduction_failed"
        next_stage = "debug the unchanged paired-prior path"
    elif not same_device:
        root = "phase314b_r254_resume5_same_device_residual_nondeterminism_supported"
        next_stage = "localize the first divergent residual step before any contract change"
    elif not resume3_exact:
        root = "phase314b_r254_resume5_isolated_worker_changed_current_device_result"
        next_stage = "audit instrumentation/process isolation before scientific inference"
    elif not functional_pass:
        root = "phase314b_r254_resume5_cable_functional_regression_supported"
        next_stage = "debug the regressed cable gate; do not relax reproduction thresholds"
    else:
        root = (
            "phase314b_r254_resume5_same_device_residual_determinism_and_"
            "cable_functional_nonregression_supported"
        )
        next_stage = (
            "Phase3.14b-r2.5.5 robot-proxy schema repair, provenance migration, "
            "and train-cache regeneration proposal"
        )
    inherited_source_audit = resume4_audit.get("robot_proxy_static_source_audit", {})
    return {
        "phase": PHASE,
        "schema": SCHEMA,
        "verdict": "PASS",
        "scientific_status": "BLOCKED",
        "root_cause": root,
        "next_stage": next_stage,
        "spec": asdict(spec),
        "repeat_comparison": repeat_comparison,
        "same_device_residual_determinism_pass": same_device,
        "fresh_exact_match_to_resume3_pass": resume3_exact,
        "functional_prior_contract_pass": prior_contract,
        "functional_cable_nonregression_pass": functional_pass,
        "functional_reproduction_contract_pass": functional_reproduction_pass,
        "historical_boolean_exact_agreement_pass": strict_boolean_exact,
        "models": model_reports,
        "repeats": list(repeats),
        "code_audit_findings": {
            "resume4_pinned_submodule_findings": inherited_source_audit,
            "additional_confirmed_findings": [
                {
                    "id": "robot_proxy_extractor_silently_pads_variable_length_fields",
                    "location": "ccda_phase3/data_io.py:42-59",
                    "effect": (
                        "missing joint/EE components become ordinary zeros without a "
                        "presence mask or exact-length rejection"
                    ),
                },
                {
                    "id": "robot_proxy_schema_pass_is_dimensional_not_semantic",
                    "location": (
                        "ccda_phase3/phase314b_r254_robot_proxy_attribution.py:300-414"
                    ),
                    "effect": (
                        "schema_contract_pass ignores structural-padding warnings and "
                        "cannot validate joint selection, EE link identity, or logger errors"
                    ),
                },
                {
                    "id": "one_step_physical_gate_mixes_cable_and_robot_quaternion",
                    "location": "ccda_phase3/phase314b_r21_geometry.py:213-272",
                    "effect": (
                        "the gate cannot be treated as cable-only while robot-proxy "
                        "orientation provenance is defective"
                    ),
                },
                {
                    "id": "legacy_npz_loader_eagerly_materializes_all_arrays",
                    "location": "ccda_phase3/phase314a_contract.py:124-132",
                    "effect": (
                        "formal_test_read=false means no formal evaluation/indexing; it "
                        "does not mean target arrays were absent from process memory"
                    ),
                },
            ],
        },
        "functional_contract_is_post_hoc_tolerance": False,
        "robot_proxy_attribution_interpretable": False,
        "robot_proxy_metrics_used_for_gate": False,
        "legacy_cache_loader_eagerly_materializes_full_npz": True,
        "validation_target_rows_indexed": False,
        "formal_target_rows_indexed": False,
        "validation_targets_used": False,
        "formal_test_read": False,
        "reverse_sampling_rerun": False,
        "formal_training": False,
        "checkpoint_saved": False,
        "weights_persisted": False,
        "prediction_tensor_persisted": False,
        "candidate_execution": False,
        "idm": False,
        "phase4": False,
        "cps": False,
        "train_only_recommendation": None,
        "selected_configuration": None,
    }


def render_markdown(report: Mapping[str, Any], static_test_count: int) -> str:
    lines = [
        "# Phase3.14b-r2.5.4 Resume5 Determinism Audit",
        "",
        f"- Audit verdict: `{report['verdict']}`",
        f"- Scientific status: `{report['scientific_status']}`",
        f"- Root cause: `{report['root_cause']}`",
        f"- Scoped static tests: `{int(static_test_count)} passed`",
        f"- Same-device residual determinism: `{str(report['same_device_residual_determinism_pass']).lower()}`",
        f"- Exact match to the prior RTX-4090 Resume3 run: `{str(report['fresh_exact_match_to_resume3_pass']).lower()}`",
        f"- Cable functional non-regression: `{str(report['functional_cable_nonregression_pass']).lower()}`",
        f"- Functional reproduction contract: `{str(report['functional_reproduction_contract_pass']).lower()}`",
        f"- Historical Boolean exact agreement: `{str(report['historical_boolean_exact_agreement_pass']).lower()}`",
        "- Robot-proxy attribution interpretable: `false`",
        "",
        "## Contract design",
        "",
        "Same-device identity is exact SHA equality over initialization, optimizer, RNG/sampler streams, histories, final residual state, and prediction. Cross-device functional reproduction introduces no new numeric tolerance: it requires all fixed cable gates to pass and forbids any historical true-to-false regression. Full-state and robot-proxy metrics remain excluded because the pinned logger schema is defective.",
        "",
        "## Per-model result",
        "",
        "| Model | repeat exact | Resume3 exact | cable non-regression | historical Boolean exact |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in DIAGNOSTIC_OBJECTIVE_NAMES:
        item = report["models"][name]
        lines.append(
            "| `{}` | {} | {} | {} | {} |".format(
                name,
                str(item["same_device_repeat_exact"]["exact"]).lower(),
                str(item["fresh_exact_match_to_resume3"]["exact"]).lower(),
                str(item["functional_cable_nonregression"]["pass"]).lower(),
                str(item["historical_boolean_exact_agreement"]["pass"]).lower(),
            )
        )
    lines.extend([
        "",
        "## Boundary",
        "",
        "The legacy loader eagerly materializes the locked NPZ, but this audit never indexes validation/formal target rows and never uses them in training or metrics. No reverse sampling, IDM, candidate execution, Phase4, or CPS was used. No checkpoint, weight tensor, prediction tensor, RNG state, or optimizer state was persisted. Robot-proxy metrics remain uninterpretable and no configuration is selected.",
        "",
        f"Next: `{report['next_stage']}`.",
        "",
    ])
    return "\n".join(lines)


__all__ = [
    "BASE_EVIDENCE_COMMIT",
    "CONTRACT_KEYS",
    "DIAGNOSTIC_OBJECTIVE_NAMES",
    "EXPECTED_CACHE_SHA256",
    "EXPECTED_CONTRACT_SHA256",
    "EXPECTED_CURRENT_PRIOR_PREDICTION_SHA256",
    "EXPECTED_CURRENT_PRIOR_STATE_SHA256",
    "EXPECTED_RESUME4_AUDIT_SHA256",
    "EXPECTED_RESUME4_MARKDOWN_SHA256",
    "EXPECTED_RESUME4_PREFLIGHT_SHA256",
    "EXPECTED_RESUME4_SUMMARY_SHA256",
    "EXPECTED_RESUME4_TEST_GATE_SHA256",
    "EXPECTED_RESUME5_TEST_COUNT",
    "EXPECTED_SCOPED_TEST_COUNT",
    "EXPECTED_SUBMODULE_COMMIT",
    "FUNCTIONAL_CABLE_KEYS",
    "IDENTITY_FIELDS",
    "NUMERIC_KEYS",
    "PHASE",
    "RESUME3_PILOT_PATH",
    "RESUME4_AUDIT_PATH",
    "RESUME4_MARKDOWN_PATH",
    "RESUME4_PREFLIGHT_PATH",
    "RESUME4_SUMMARY_PATH",
    "RESUME4_TEST_GATE_PATH",
    "RESUME5_TEST_PATH",
    "Resume5Spec",
    "SOURCE_PATHS",
    "assert_json_summary_safe",
    "assert_only_allowed_worktree_paths",
    "build_resume5_audit",
    "canonical_sha256",
    "compare_fresh_to_resume3",
    "compare_repeat_matrix",
    "functional_reproduction_contract",
    "git_output",
    "instrument_residual_training",
    "load_json",
    "module_state_sha256",
    "optimizer_state_sha256",
    "parse_worker_stdout",
    "render_markdown",
    "sha256_file",
    "source_sha256",
    "tensor_fingerprint",
    "write_json_once",
    "write_text_once",
]

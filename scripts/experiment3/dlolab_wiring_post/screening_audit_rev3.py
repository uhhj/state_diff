"""PB0-S REV3: offline screening-assumption audit for DLO-Lab Wiring-post.

Scientific boundaries:
- formal PB0 verdict is unchanged;
- no new simulation/model training;
- each future horizon uses its own valid time range;
- ordered indexed RMSE is diagnostic only;
- Chamfer is audited by an independent, positive-evidence-only curve metric;
- cross-time zero candidates are non-conclusive;
- "peak horizon" is never claimed: only the maximum among sampled horizons.
"""
from __future__ import annotations

import argparse
import heapq
import json
from itertools import combinations
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiment3.dlolab_wiring_post.features import (  # noqa: E402
    hidden_difference,
    oracle_descriptor,
    post_visible_mask,
    symmetric_chamfer,
    visible_points,
)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git(*args):
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def stats(values):
    if not values:
        return {
            "n": 0,
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    x = np.asarray(values, dtype=np.float64)
    return {
        "n": int(x.size),
        "min": float(np.min(x)),
        "p05": float(np.percentile(x, 5)),
        "median": float(np.percentile(x, 50)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def history_ee_distance(a, b):
    a = np.asarray(a, dtype=np.float64)[..., :3]
    b = np.asarray(b, dtype=np.float64)[..., :3]
    return float(np.linalg.norm(a - b, axis=1).mean())


def history_chamfer(a, b):
    return float(np.mean([
        symmetric_chamfer(x, y)
        for x, y in zip(a, b)
    ]))


def history_chamfer_if_pass(a, b, threshold):
    threshold = float(threshold)
    total = 0.0
    n = len(a)
    for x, y in zip(a, b):
        total += symmetric_chamfer(x, y)
        if total > threshold * n:
            return None
    return float(total / n)


def global_open_rope_angle(rope_xyz, post_xy):
    rope_xy = np.asarray(rope_xyz, dtype=np.float64)[:, :2]
    center = np.asarray(post_xy, dtype=np.float64)
    rel = rope_xy - center[None, :]
    first = rel[:-1]
    second = rel[1:]
    cross = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    dot = np.sum(first * second, axis=1)
    return float(np.arctan2(cross, dot).sum())


def descriptor_diagnostic(
        rope_a,
        rope_b,
        post_xy,
        oracle_cfg,
        global_threshold):
    desc_a = oracle_descriptor(
        rope_a,
        post_xy,
        rope_radius_m=oracle_cfg["rope_radius_m"],
        post_radius_m=oracle_cfg["post_radius_m"],
        roi_radius_m=oracle_cfg["post_roi_radius_m"],
        contact_proxy_margin_m=oracle_cfg["contact_proxy_margin_m"],
    )
    desc_b = oracle_descriptor(
        rope_b,
        post_xy,
        rope_radius_m=oracle_cfg["rope_radius_m"],
        post_radius_m=oracle_cfg["post_radius_m"],
        roi_radius_m=oracle_cfg["post_roi_radius_m"],
        contact_proxy_margin_m=oracle_cfg["contact_proxy_margin_m"],
    )
    original = hidden_difference(
        desc_a,
        desc_b,
        oracle_cfg["wrap_difference_min_rad"],
    )
    global_a = [
        global_open_rope_angle(rope_a, center)
        for center in np.asarray(post_xy)
    ]
    global_b = [
        global_open_rope_angle(rope_b, center)
        for center in np.asarray(post_xy)
    ]
    global_diff = [
        abs(float(x) - float(y))
        for x, y in zip(global_a, global_b)
    ]
    global_diff_flag = bool(
        original["contact_like_mismatch"]
        or max(global_diff, default=0.0) >= float(global_threshold)
    )
    return {
        "original_hidden_different": bool(original["different"]),
        "contact_like_mismatch": bool(original["contact_like_mismatch"]),
        "roi_wrap_difference_rad": [
            float(v) for v in original["wrap_difference_rad"]
        ],
        "global_angular_difference_rad": [
            float(v) for v in global_diff
        ],
        "global_hidden_different_diagnostic": global_diff_flag,
        "descriptor_a": desc_a,
        "descriptor_b": desc_b,
    }


def ordered_visibility_diagnostic(
        rope_history_a,
        rope_history_b,
        post_xy,
        radius):
    rmse = []
    mismatch = []
    common_counts = []
    only_a_counts = []
    only_b_counts = []

    for a, b in zip(
            np.asarray(rope_history_a, dtype=np.float64),
            np.asarray(rope_history_b, dtype=np.float64)):
        ma = post_visible_mask(a, post_xy, radius)
        mb = post_visible_mask(b, post_xy, radius)
        common = ma & mb
        mismatch.append(float(np.mean(ma != mb)))
        common_counts.append(int(np.sum(common)))
        only_a_counts.append(int(np.sum(ma & ~mb)))
        only_b_counts.append(int(np.sum(mb & ~ma)))
        if np.any(common):
            d = a[common] - b[common]
            rmse.append(float(np.sqrt(np.mean(np.sum(d * d, axis=1)))))

    return {
        "common_visible_indexed_rmse_m":
            float(np.mean(rmse)) if rmse else None,
        "visible_mask_mismatch_fraction":
            float(np.mean(mismatch)),
        "common_visible_vertex_count_mean":
            float(np.mean(common_counts)),
        "only_a_visible_vertex_count_mean":
            float(np.mean(only_a_counts)),
        "only_b_visible_vertex_count_mean":
            float(np.mean(only_b_counts)),
    }


def point_to_segment_distance(point, start, end):
    point = np.asarray(point, dtype=np.float64)
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    edge = end - start
    denom = float(np.dot(edge, edge))
    if denom <= 1e-18:
        return float(np.linalg.norm(point - start))
    alpha = float(np.dot(point - start, edge) / denom)
    alpha = min(1.0, max(0.0, alpha))
    projection = start + alpha * edge
    return float(np.linalg.norm(point - projection))


def visible_polyline_segments(rope_xyz, visible_mask):
    rope_xyz = np.asarray(rope_xyz, dtype=np.float64)
    mask = np.asarray(visible_mask, dtype=bool)
    return [
        (rope_xyz[i], rope_xyz[i + 1])
        for i in range(len(rope_xyz) - 1)
        if mask[i] and mask[i + 1]
    ]


def mean_points_to_segments(points, segments):
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0 or len(segments) == 0:
        return float("inf")
    values = []
    for point in points:
        values.append(min(
            point_to_segment_distance(point, start, end)
            for start, end in segments
        ))
    return float(np.mean(values))


def symmetric_visible_curve_distance(
        rope_a,
        rope_b,
        post_xy,
        radius):
    """Independent curve-geometry diagnostic for sparse-vertex Chamfer.

    It compares visible vertices of one rope to visible continuous polyline
    segments of the other rope. It never becomes a formal admission metric.
    """
    ma = post_visible_mask(rope_a, post_xy, radius)
    mb = post_visible_mask(rope_b, post_xy, radius)
    points_a = np.asarray(rope_a, dtype=np.float64)[ma]
    points_b = np.asarray(rope_b, dtype=np.float64)[mb]
    seg_a = visible_polyline_segments(rope_a, ma)
    seg_b = visible_polyline_segments(rope_b, mb)
    ab = mean_points_to_segments(points_a, seg_b)
    ba = mean_points_to_segments(points_b, seg_a)
    return float(0.5 * (ab + ba)), ma, mb


def masked_signature(
        rope_history,
        ee_history,
        post_xy,
        radius,
        key_vertices):
    values = []
    key_vertices = [int(v) for v in key_vertices]

    for frame in np.asarray(rope_history, dtype=np.float64):
        mask = post_visible_mask(frame, post_xy, radius)
        for vertex in key_vertices:
            if mask[vertex]:
                values.extend(frame[vertex].tolist())
                values.append(1.0)
            else:
                nearest = int(np.argmin(np.linalg.norm(
                    post_xy - frame[vertex, :2][None, :],
                    axis=1,
                )))
                values.extend([
                    float(post_xy[nearest, 0]),
                    float(post_xy[nearest, 1]),
                    0.0,
                ])
                values.append(0.0)

    values.extend(
        np.asarray(ee_history, dtype=np.float64)[..., :3]
        .reshape(-1)
        .tolist()
    )
    return np.asarray(values, dtype=np.float32)


def push_smallest(heap, value, serial, row, top_k):
    item = (-float(value), int(serial), row)
    if len(heap) < int(top_k):
        heapq.heappush(heap, item)
        return
    if float(value) < -heap[0][0]:
        heapq.heapreplace(heap, item)


def unpack_smallest(heap):
    rows = [(-item[0], item[2]) for item in heap]
    rows.sort(key=lambda x: x[0])
    return [row for _, row in rows]


def visible_history_for_pair(
        rope,
        rollout,
        time_index,
        history,
        post_xy,
        radius):
    start = time_index - history + 1
    return [
        visible_points(
            rope[rollout, frame],
            post_xy,
            radius,
        )
        for frame in range(start, time_index + 1)
    ]


def scan_observation_hidden_pairs(
        data,
        pb0_cfg,
        audit_cfg):
    """Single full-time scan.

    It records all original-observable+EE pairs and all hidden-different pairs.
    Formal PB0 funnel counts are incremented only when t+10 exists, exactly
    reproducing the original PB0 eligible time range.
    """
    rope = data["rope_xyz"]
    ee = data["ee_pos"]
    post_xyz = data["post_xyz"]

    obs = pb0_cfg["observation_protocol"]
    oracle = pb0_cfg["oracle_pair_descriptor"]
    mining = pb0_cfg["pair_mining"]

    history = int(obs["history_samples"])
    radius = float(obs["post_occlusion_radius_m"])
    obs_threshold = float(mining["max_observable_chamfer_m"])
    ee_threshold = float(mining["max_ee_position_distance_m"])
    original_h = int(mining["future_horizon_samples"])
    ratio_threshold = float(mining["min_future_to_observable_ratio"])
    global_threshold = float(
        audit_cfg["descriptor_diagnostics"][
            "global_angular_difference_min_rad"
        ]
    )
    top_k = 20

    counts = {
        "formal_total_same_time_pairs": 0,
        "formal_chamfer_pass": 0,
        "formal_ee_pass_after_chamfer": 0,
        "formal_original_hidden_pass": 0,
        "formal_t10_ratio_pass": 0,
        "all_time_original_hidden_pair_count": 0,
        "all_time_global_only_pair_count": 0,
    }

    original_hidden_pairs = []
    global_only_pairs = []
    top_original_hidden = []
    top_global_only = []
    serial = 0

    for t in range(history - 1, rope.shape[1]):
        formal_time = (t + original_h < rope.shape[1])
        start = t - history + 1

        for a, b in combinations(range(rope.shape[0]), 2):
            if formal_time:
                counts["formal_total_same_time_pairs"] += 1

            post_xy = 0.5 * (
                post_xyz[a, t, :, :2]
                + post_xyz[b, t, :, :2]
            )

            vis_a = visible_history_for_pair(
                rope, a, t, history, post_xy, radius
            )
            vis_b = visible_history_for_pair(
                rope, b, t, history, post_xy, radius
            )
            chamfer = history_chamfer_if_pass(
                vis_a, vis_b, obs_threshold
            )
            if chamfer is None:
                continue

            if formal_time:
                counts["formal_chamfer_pass"] += 1

            ee_distance = history_ee_distance(
                ee[a, start:t + 1],
                ee[b, start:t + 1],
            )
            if ee_distance > ee_threshold:
                continue

            if formal_time:
                counts["formal_ee_pass_after_chamfer"] += 1

            descriptor = descriptor_diagnostic(
                rope[a, t],
                rope[b, t],
                post_xy,
                oracle,
                global_threshold,
            )
            ordered = ordered_visibility_diagnostic(
                rope[a, start:t + 1],
                rope[b, start:t + 1],
                post_xy,
                radius,
            )

            base = {
                "rollout_a": int(a),
                "rollout_b": int(b),
                "time_index": int(t),
                "observable_chamfer_m": float(chamfer),
                "ee_position_distance_m": float(ee_distance),
                "ordered_visibility_diagnostic": ordered,
                "hidden_descriptor_diagnostic": descriptor,
            }

            if descriptor["original_hidden_different"]:
                counts["all_time_original_hidden_pair_count"] += 1
                original_hidden_pairs.append(base)
                serial += 1
                push_smallest(
                    top_original_hidden,
                    chamfer,
                    serial,
                    base,
                    top_k,
                )

                if formal_time:
                    counts["formal_original_hidden_pass"] += 1
                    future_a = visible_points(
                        rope[a, t + original_h], post_xy, radius
                    )
                    future_b = visible_points(
                        rope[b, t + original_h], post_xy, radius
                    )
                    future = symmetric_chamfer(future_a, future_b)
                    ratio = float(future / max(chamfer, 1e-5))
                    if ratio >= ratio_threshold:
                        counts["formal_t10_ratio_pass"] += 1

            elif descriptor[
                    "global_hidden_different_diagnostic"]:
                counts["all_time_global_only_pair_count"] += 1
                global_only_pairs.append(base)
                serial += 1
                push_smallest(
                    top_global_only,
                    chamfer,
                    serial,
                    base,
                    top_k,
                )

    return {
        "counts": counts,
        "original_hidden_pairs": original_hidden_pairs,
        "global_only_pairs": global_only_pairs,
        "top_original_hidden_pairs": unpack_smallest(top_original_hidden),
        "top_global_only_pairs": unpack_smallest(top_global_only),
    }


def horizon_profile_for_pairs(
        rope,
        post_xyz,
        pairs,
        pb0_cfg,
        horizons):
    radius = float(
        pb0_cfg["observation_protocol"]["post_occlusion_radius_m"]
    )

    per_horizon = {
        str(int(h)): []
        for h in horizons
    }
    eligible_counts = {
        str(int(h)): 0
        for h in horizons
    }
    max_sampled_horizon_counts = {}
    max_sampled_over_t10 = []

    for row in pairs:
        a = int(row["rollout_a"])
        b = int(row["rollout_b"])
        t = int(row["time_index"])
        post_xy = 0.5 * (
            post_xyz[a, t, :, :2]
            + post_xyz[b, t, :, :2]
        )

        available = {}
        for h in horizons:
            h = int(h)
            if t + h >= rope.shape[1]:
                continue
            eligible_counts[str(h)] += 1
            fa = visible_points(rope[a, t + h], post_xy, radius)
            fb = visible_points(rope[b, t + h], post_xy, radius)
            distance = float(symmetric_chamfer(fa, fb))
            per_horizon[str(h)].append(distance)
            available[h] = distance

        if not available:
            continue

        h_max = max(
            available,
            key=lambda h: available[h],
        )
        max_sampled_horizon_counts[str(int(h_max))] = (
            max_sampled_horizon_counts.get(str(int(h_max)), 0)
            + 1
        )

        if 10 in available:
            max_sampled_over_t10.append(
                float(
                    available[h_max]
                    / max(available[10], 1e-9)
                )
            )

    return {
        "horizon_eligible_pair_count": eligible_counts,
        "future_distance_summary_by_horizon_m": {
            h: stats(values)
            for h, values in per_horizon.items()
        },
        "horizon_of_maximum_sampled_future_distance_counts":
            max_sampled_horizon_counts,
        "maximum_sampled_future_distance_over_t10_ratio":
            stats(max_sampled_over_t10),
        "naming_note":
            (
                "The reported horizon is only the horizon with the maximum "
                "distance among the sampled horizons available for that pair. "
                "It is not claimed to be the physical peak-divergence time."
            ),
    }


def same_time_chamfer_metric_audit(
        data,
        pb0_cfg,
        audit_cfg):
    """Positive-evidence-only audit of sparse-vertex Chamfer.

    Proposals come from an independent masked fixed-length signature. Exact
    curve distance, EE, visibility-mask equality, hidden descriptor and the
    original vertex Chamfer are then recomputed.

    A positive hit means a plausible pair is curve-near yet rejected by the
    original vertex Chamfer. A zero hit is non-conclusive because kNN proposal
    search is not exhaustive.
    """
    cfg = audit_cfg["chamfer_metric_audit"]
    if not cfg["enabled"]:
        return {"enabled": False}

    rope = data["rope_xyz"]
    ee = data["ee_pos"]
    post_xyz = data["post_xyz"]

    obs = pb0_cfg["observation_protocol"]
    oracle = pb0_cfg["oracle_pair_descriptor"]
    mining = pb0_cfg["pair_mining"]

    history = int(obs["history_samples"])
    radius = float(obs["post_occlusion_radius_m"])
    vertex_chamfer_threshold = float(
        mining["max_observable_chamfer_m"]
    )
    ee_threshold = float(
        mining["max_ee_position_distance_m"]
    )
    curve_threshold = float(
        cfg["curve_distance_threshold_m"]
    )
    global_threshold = float(
        audit_cfg["descriptor_diagnostics"][
            "global_angular_difference_min_rad"
        ]
    )
    keys = audit_cfg["diagnostic_signature"]["key_vertices"]
    k = int(cfg["knn_k"])
    require_same_mask = bool(
        cfg["require_identical_visibility_mask"]
    )
    top_k = int(cfg["top_k"])

    positives = []
    examined = 0

    canonical_post_xy = np.mean(
        post_xyz[..., :2],
        axis=(0, 1),
    )

    for t in range(history - 1, rope.shape[1]):
        start = t - history + 1
        signatures = np.stack([
            masked_signature(
                rope[r, start:t + 1],
                ee[r, start:t + 1],
                canonical_post_xy,
                radius,
                keys,
            )
            for r in range(rope.shape[0])
        ], axis=0)

        tree = cKDTree(signatures)
        query_k = min(k + 1, rope.shape[0])
        _, nn = tree.query(
            signatures,
            k=query_k,
            workers=-1,
        )

        seen = set()
        for r, neighbors in enumerate(np.atleast_2d(nn)):
            for other in np.atleast_1d(neighbors):
                other = int(other)
                if other == r:
                    continue
                pair = tuple(sorted((r, other)))
                if pair in seen:
                    continue
                seen.add(pair)
                examined += 1
                a, b = pair

                post_xy = 0.5 * (
                    post_xyz[a, t, :, :2]
                    + post_xyz[b, t, :, :2]
                )

                ee_distance = history_ee_distance(
                    ee[a, start:t + 1],
                    ee[b, start:t + 1],
                )
                if ee_distance > ee_threshold:
                    continue

                curve_values = []
                masks_identical = True
                for frame in range(start, t + 1):
                    curve_distance, ma, mb = (
                        symmetric_visible_curve_distance(
                            rope[a, frame],
                            rope[b, frame],
                            post_xy,
                            radius,
                        )
                    )
                    curve_values.append(curve_distance)
                    masks_identical = (
                        masks_identical
                        and bool(np.array_equal(ma, mb))
                    )

                curve_distance = float(np.mean(curve_values))
                if curve_distance > curve_threshold:
                    continue
                if require_same_mask and not masks_identical:
                    continue

                vis_a = visible_history_for_pair(
                    rope, a, t, history, post_xy, radius
                )
                vis_b = visible_history_for_pair(
                    rope, b, t, history, post_xy, radius
                )
                vertex_chamfer = history_chamfer(vis_a, vis_b)

                if vertex_chamfer <= vertex_chamfer_threshold:
                    continue

                descriptor = descriptor_diagnostic(
                    rope[a, t],
                    rope[b, t],
                    post_xy,
                    oracle,
                    global_threshold,
                )
                if not (
                    descriptor["original_hidden_different"]
                    or descriptor[
                        "global_hidden_different_diagnostic"
                    ]
                ):
                    continue

                positives.append({
                    "rollout_a": int(a),
                    "rollout_b": int(b),
                    "time_index": int(t),
                    "curve_distance_m": curve_distance,
                    "original_vertex_chamfer_m": vertex_chamfer,
                    "ee_position_distance_m": ee_distance,
                    "visibility_masks_identical": masks_identical,
                    "hidden_descriptor_diagnostic": descriptor,
                })

    positives.sort(
        key=lambda row: (
            row["curve_distance_m"],
            row["original_vertex_chamfer_m"],
        )
    )

    return {
        "enabled": True,
        "search_is_exhaustive": False,
        "proposal_method":
            "same-time cKDTree masked-history signature",
        "knn_k": k,
        "proposal_pairs_examined": int(examined),
        "positive_candidate_count": int(len(positives)),
        "top_candidates": positives[:top_k],
        "interpretation": (
            "Positive candidates are evidence that sparse-vertex Chamfer may "
            "reject curve-near states. Zero candidates are non-conclusive "
            "because the proposal search is approximate."
        ),
    }


def past_action_delta_rmse(
        common_qpos,
        time_a,
        time_b,
        history):
    deltas = np.diff(
        np.asarray(common_qpos, dtype=np.float64),
        axis=0,
    )
    first = deltas[time_a - history:time_a]
    second = deltas[time_b - history:time_b]
    return float(np.sqrt(np.mean((first - second) ** 2)))


def cross_time_audit(
        data,
        pb0_cfg,
        audit_cfg):
    cfg = audit_cfg["cross_time_audit"]
    if not cfg["enabled"]:
        return {"enabled": False}

    rope = data["rope_xyz"]
    ee = data["ee_pos"]
    post_xyz = data["post_xyz"]
    qpos = data["common_qpos_replay"]

    obs = pb0_cfg["observation_protocol"]
    oracle = pb0_cfg["oracle_pair_descriptor"]
    mining = pb0_cfg["pair_mining"]

    history = int(obs["history_samples"])
    action_history = int(cfg["action_history_samples"])
    radius = float(obs["post_occlusion_radius_m"])
    obs_threshold = float(mining["max_observable_chamfer_m"])
    ee_threshold = float(mining["max_ee_position_distance_m"])
    global_threshold = float(
        audit_cfg["descriptor_diagnostics"][
            "global_angular_difference_min_rad"
        ]
    )
    min_dt = int(cfg["min_time_delta_samples"])
    keys = audit_cfg["diagnostic_signature"]["key_vertices"]

    canonical_post_xy = np.mean(
        post_xyz[..., :2],
        axis=(0, 1),
    )

    start_time = max(history - 1, action_history)
    states = []
    signatures = []

    for r in range(rope.shape[0]):
        for t in range(start_time, rope.shape[1]):
            start = t - history + 1
            states.append((int(r), int(t)))
            signatures.append(
                masked_signature(
                    rope[r, start:t + 1],
                    ee[r, start:t + 1],
                    canonical_post_xy,
                    radius,
                    keys,
                )
            )

    signatures = np.stack(signatures, axis=0)
    tree = cKDTree(signatures)
    k = min(int(cfg["knn_k"]) + 1, len(states))
    _, nn = tree.query(signatures, k=k, workers=-1)

    seen = set()
    positives = []

    for state_id, neighbors in enumerate(np.atleast_2d(nn)):
        ra, ta = states[state_id]
        for neighbor in np.atleast_1d(neighbors):
            neighbor = int(neighbor)
            if neighbor == state_id:
                continue
            key = tuple(sorted((state_id, neighbor)))
            if key in seen:
                continue
            seen.add(key)

            rb, tb = states[neighbor]
            if abs(ta - tb) < min_dt:
                continue

            start_a = ta - history + 1
            start_b = tb - history + 1
            post_xy = 0.5 * (
                post_xyz[ra, ta, :, :2]
                + post_xyz[rb, tb, :, :2]
            )

            vis_a = [
                visible_points(frame, post_xy, radius)
                for frame in rope[ra, start_a:ta + 1]
            ]
            vis_b = [
                visible_points(frame, post_xy, radius)
                for frame in rope[rb, start_b:tb + 1]
            ]
            chamfer = history_chamfer_if_pass(
                vis_a, vis_b, obs_threshold
            )
            if chamfer is None:
                continue

            ee_distance = history_ee_distance(
                ee[ra, start_a:ta + 1],
                ee[rb, start_b:tb + 1],
            )
            if ee_distance > ee_threshold:
                continue

            descriptor = descriptor_diagnostic(
                rope[ra, ta],
                rope[rb, tb],
                post_xy,
                oracle,
                global_threshold,
            )
            if not (
                descriptor["original_hidden_different"]
                or descriptor[
                    "global_hidden_different_diagnostic"
                ]
            ):
                continue

            positives.append({
                "rollout_a": int(ra),
                "time_a": int(ta),
                "rollout_b": int(rb),
                "time_b": int(tb),
                "time_delta_samples": int(abs(ta - tb)),
                "observable_chamfer_m": float(chamfer),
                "ee_position_distance_m": float(ee_distance),
                "past_action_delta_qpos_rmse": past_action_delta_rmse(
                    qpos, ta, tb, action_history
                ),
                "hidden_descriptor_diagnostic": descriptor,
            })

    positives.sort(key=lambda row: (
        row["observable_chamfer_m"],
        row["ee_position_distance_m"],
        row["past_action_delta_qpos_rmse"],
    ))

    return {
        "enabled": True,
        "search_is_exhaustive": False,
        "knn_k": int(cfg["knn_k"]),
        "states_indexed": int(len(states)),
        "unique_knn_pairs_examined": int(len(seen)),
        "positive_candidate_count": int(len(positives)),
        "top_candidates": positives[:int(cfg["top_k"])],
        "interpretation": (
            "Positive candidates support the possibility that the same-time "
            "restriction misses plausible pairs. Zero candidates are not "
            "negative evidence because this search is approximate."
        ),
    }


def findings(
        scan,
        chamfer_audit,
        cross_time):
    counts = scan["counts"]
    found = ["PB0S_ORIGINAL_FUNNEL_CHARACTERIZED"]

    if counts["formal_original_hidden_pass"] > 0:
        found.append(
            "PB0S_ORIGINAL_HIDDEN_PAIRS_EXIST_BEFORE_T10_FILTER"
        )

    if counts["all_time_global_only_pair_count"] > 0:
        found.append(
            "PB0S_ALTERNATIVE_DESCRIPTOR_REVEALS_ADDITIONAL_ROUTING_VARIATION"
        )

    if chamfer_audit.get("positive_candidate_count", 0) > 0:
        found.append(
            "PB0S_VERTEX_CHAMFER_FALSE_NEGATIVE_PLAUSIBLE"
        )

    if cross_time.get("positive_candidate_count", 0) > 0:
        found.append(
            "PB0S_SAME_TIME_RESTRICTION_FALSE_NEGATIVE_PLAUSIBLE"
        )

    if len(found) == 1:
        found.append(
            "PB0S_NO_SCREENING_FALSE_NEGATIVE_DEMONSTRATED"
        )

    return found


def next_action(found):
    if "PB0S_ORIGINAL_HIDDEN_PAIRS_EXIST_BEFORE_T10_FILTER" in found:
        return (
            "Pre-register snapshot same-action branching on top original-"
            "descriptor pairs. Evaluate the future at multiple horizons and "
            "compare branch divergence against deterministic repeat/uncertainty "
            "at each horizon. Do not reuse future/current>=2 as formal Gate 4."
        )
    if "PB0S_VERTEX_CHAMFER_FALSE_NEGATIVE_PLAUSIBLE" in found:
        return (
            "Pre-register the curve-based observable metric and validate it "
            "on held-out Wiring-post rollouts before using it for formal pair "
            "admission."
        )
    if "PB0S_ALTERNATIVE_DESCRIPTOR_REVEALS_ADDITIONAL_ROUTING_VARIATION" in found:
        return (
            "Pre-register the revised hidden-routing descriptor and validate "
            "it on held-out Wiring-post data before formal pair admission."
        )
    if "PB0S_SAME_TIME_RESTRICTION_FALSE_NEGATIVE_PLAUSIBLE" in found:
        return (
            "Pre-register action-history-aware cross-time discovery on held-"
            "out data, then use snapshot restore to impose identical future "
            "actions."
        )
    return (
        "No specific false-negative mechanism was positively demonstrated in "
        "the frozen dataset. This does not prove absence of CCDA. For project "
        "efficiency, choose one pre-registered diverse-behavior Wiring-post "
        "collection or move to the next published task."
    )


def build_report(
        audit_cfg,
        scan,
        original_horizon,
        global_horizon,
        chamfer_audit,
        cross_time):
    found = findings(scan, chamfer_audit, cross_time)
    action = next_action(found)

    result = {
        "verdict": "PB0S_REV3_SCREENING_ASSUMPTIONS_AUDITED",
        "formal_pb0_verdict":
            audit_cfg["source"]["formal_pb0_verdict"],
        "formal_pb0_verdict_changed": False,
        "new_simulator_rollout": False,
        "model_training_started": False,
        "findings": found,
        "formal_funnel_counts": scan["counts"],
        "original_hidden_horizon_audit": original_horizon,
        "global_only_horizon_audit": global_horizon,
        "chamfer_metric_audit": chamfer_audit,
        "cross_time_audit": cross_time,
        "top_original_hidden_pairs":
            scan["top_original_hidden_pairs"],
        "top_global_only_pairs":
            scan["top_global_only_pairs"],
        "next_action": action,
        "non_claims": [
            "Zero hidden-descriptor hits do not imply a hidden single mode.",
            "Ordered indexed RMSE never admits a pair.",
            "Zero same-time Chamfer-audit hits are non-conclusive.",
            "Zero cross-time hits are non-conclusive.",
            "The horizon of maximum sampled future distance is not a claimed physical peak time.",
            "Multi-horizon future distance is diagnostic until compared against a repeat/uncertainty floor.",
        ],
    }

    raw_root = Path(audit_cfg["outputs"]["raw_root"])
    raw_root.mkdir(parents=True, exist_ok=True)
    (raw_root / "SCREENING_ASSUMPTION_AUDIT.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )

    committed = (
        REPO_ROOT
        / audit_cfg["outputs"]["committed_report_dir"]
    )
    committed.mkdir(parents=True, exist_ok=True)

    evidence = {
        "phase_name": audit_cfg["phase_name"],
        "verdict": result["verdict"],
        "repository": {
            "starting_main_sha":
                audit_cfg["provenance"]["starting_main_sha"],
            "ending_main_sha": git("rev-parse", "HEAD"),
            "dlolab_gitlink":
                git("rev-parse", "HEAD:external/dlo-lab"),
        },
        "scientific": result,
    }
    (committed / "EVIDENCE.json").write_text(
        json.dumps(evidence, indent=2) + "\n",
        encoding="utf-8",
    )

    c = scan["counts"]
    lines = [
        "# PB0-S REV3 Wiring-post Screening Assumption Audit",
        "",
        "Verdict: `PB0S_REV3_SCREENING_ASSUMPTIONS_AUDITED`",
        "",
        "Formal PB0 verdict remains: `{}`".format(
            result["formal_pb0_verdict"]
        ),
        "",
        "## Formal PB0 funnel",
        "",
        "- t+10-eligible same-time pairs: {}".format(
            c["formal_total_same_time_pairs"]
        ),
        "- Chamfer pass: {}".format(c["formal_chamfer_pass"]),
        "- EE pass: {}".format(c["formal_ee_pass_after_chamfer"]),
        "- Original hidden pass: {}".format(
            c["formal_original_hidden_pass"]
        ),
        "- Original t+10 ratio pass: {}".format(
            c["formal_t10_ratio_pass"]
        ),
        "",
        "## Full-time hidden-pair discovery for horizon audit",
        "",
        "- Original-hidden pairs across all current-state times: {}".format(
            c["all_time_original_hidden_pair_count"]
        ),
        "- Global-descriptor-only pairs: {}".format(
            c["all_time_global_only_pair_count"]
        ),
        "",
        "Each horizon uses its own eligible current-state time range.",
        "",
        "## Chamfer metric audit",
        "",
        "- Search exhaustive: {}".format(
            chamfer_audit.get("search_is_exhaustive")
        ),
        "- Positive candidate count: {}".format(
            chamfer_audit.get("positive_candidate_count", 0)
        ),
        "- Zero is non-conclusive.",
        "",
        "## Cross-time audit",
        "",
        "- Search exhaustive: {}".format(
            cross_time.get("search_is_exhaustive")
        ),
        "- Positive candidate count: {}".format(
            cross_time.get("positive_candidate_count", 0)
        ),
        "- Zero is non-conclusive.",
        "",
        "## Findings",
        "",
    ]
    lines.extend("- `{}`".format(x) for x in found)
    lines.extend([
        "",
        "## Next action",
        "",
        action,
        "",
        "## Terminology",
        "",
        "REV3 reports `horizon_of_maximum_sampled_future_distance`; "
        "it does not use or claim a physical `peak horizon`.",
        "",
    ])
    (committed / "RESULT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return result


def run(config_path):
    audit_cfg = load_json(config_path)
    pb0_cfg = load_json(
        REPO_ROOT / audit_cfg["source"]["pb0_config"]
    )
    dataset_path = Path(audit_cfg["source"]["dataset"])
    if not dataset_path.is_file():
        raise FileNotFoundError(str(dataset_path))

    with np.load(dataset_path) as data:
        scan = scan_observation_hidden_pairs(
            data,
            pb0_cfg,
            audit_cfg,
        )

        reproduced = int(
            scan["counts"]["formal_t10_ratio_pass"]
        )
        expected = int(
            audit_cfg["source"]["expected_formal_candidate_count"]
        )
        if reproduced != expected:
            raise RuntimeError(
                "Formal PB0 candidate count was not reproduced: "
                "{} != {}".format(reproduced, expected)
            )

        horizons = [
            int(h)
            for h in audit_cfg["horizon_audit"][
                "horizons_samples"
            ]
        ]

        original_horizon = horizon_profile_for_pairs(
            data["rope_xyz"],
            data["post_xyz"],
            scan["original_hidden_pairs"],
            pb0_cfg,
            horizons,
        )
        global_horizon = horizon_profile_for_pairs(
            data["rope_xyz"],
            data["post_xyz"],
            scan["global_only_pairs"],
            pb0_cfg,
            horizons,
        )

        chamfer_audit = same_time_chamfer_metric_audit(
            data,
            pb0_cfg,
            audit_cfg,
        )
        cross_time = cross_time_audit(
            data,
            pb0_cfg,
            audit_cfg,
        )

    return build_report(
        audit_cfg,
        scan,
        original_horizon,
        global_horizon,
        chamfer_audit,
        cross_time,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run(args.config)
    print("verdict={}".format(result["verdict"]))
    for item in result["findings"]:
        print("finding={}".format(item))


if __name__ == "__main__":
    main()

# Phase2.5b Code Audit

- Timestamp: `2026-07-05T12:07:38+08:00`
- Main HEAD: `ded17b1b667d52e5157bc93e746153a134e2c1e1`
- Submodule HEAD: `90c0b842f86ef4a9de0101e5e3d5a14a9db8cde7`

## Fixes Verified

- Hidden-condition `nominal` now replays the paired `free` oracle action plan and does not call the hidden-state oracle online.
- `oracle_breakaway_then_place` performs a release pull followed by endpoint placement.
- `hidden_breakaway_pin` records release status, release step, and max displacement.
- `guided_search` is included for free-branch search sanity.
- The selector records `selected_recoverable_config` and its env vars.

## Current Phase2.5b Result

- Verdict: `PASS`
- Selected condition: `hidden_breakaway_pin`
- Selected config: `breakaway_force_2p6_disp_0p045_pull_0p36`
- Selected env: `{"CCDA_BREAKAWAY_BEAD_RATIO": "0.45", "CCDA_BREAKAWAY_DISP": "0.045", "CCDA_BREAKAWAY_FORCE": "2.6", "CCDA_ORACLE_BREAKAWAY_PULL_DIST": "0.36"}`
- Search sanity: `{'free_guided_search_success': 1.0, 'free_nominal_success': 1.0, 'search_sanity_pass': True}`

## recoverability_audit_grep

```text
29:    "oracle_breakaway_then_place",
32:SEARCH_POLICIES = {"random_search", "guided_search", "cem_search"}
276:    if policy == "oracle_breakaway_then_place":
323:    if policy == "guided_search":
324:        return {"policy": "guided_search", "pull_dist": float(rng.uniform(0.10, 0.24)), "side_dist": float(rng.uniform(-0.08, 0.08)), "endpoint_jitter": float(rng.uniform(-0.04, 0.04)), "candidate_id": int(candidate_id)}
333:    if spec.get("policy") == "guided_search" and getattr(task, "hidden_condition", "") == "free":
340:    if spec.get("policy") == "guided_search":
376:    return {"condition": condition, "visible_seed": int(seed), "policy": policy, "trial_id": int(trial_id), "success": bool(succ), "final_fraction": float(frac), "final_curve": curve_score(final_xy), "final_chamfer_to_free": float("nan"), "final_bead_xy": final_xy.tolist(), "num_steps": int(len(actions) if max_steps else 0), "branch_label": condition, "recoverability_params": meta.get("recoverability_params", {}), "hidden_contact_meta": meta, "breakaway_released": bool(meta.get("breakaway_released", False)), "breakaway_release_step": meta.get("breakaway_release_step"), "breakaway_max_disp_seen": safe_float(meta.get("breakaway_max_disp_seen"), 0.0), "actions": actions, "total_reward": float(total_reward)}
380:def collect_free_nominal_plan(env, task_name: str, seed: int, pair_group: str, max_steps: int) -> List[Dict[str, Any]]:
415:        nominal_plan = collect_free_nominal_plan(env, task_name, seed, pair_group, max_steps)
528:    if policy == "guided_search":
529:        spec = candidate_policy_sequence("guided_search", 0, rng)
562:            release_steps = [safe_float(r.get("breakaway_release_step"), float("nan")) for r in rs if r.get("breakaway_released")]
563:            item[policy] = {"count": len(rs), "success_rate": float(np.mean([1.0 if r["success"] else 0.0 for r in rs])), "final_fraction_mean": float(np.mean([safe_float(r["final_fraction"], 0.0) for r in rs])), "final_curve_mean": float(np.mean([safe_float(r["final_curve"], 0.0) for r in rs])), "future_divergence_vs_free_mean": float(np.mean(div_vals)) if div_vals else float("nan"), "breakaway_released_rate": float(np.mean([1.0 if r.get("breakaway_released") else 0.0 for r in rs])), "breakaway_release_step_mean": float(np.mean(release_steps)) if release_steps else float("nan"), "breakaway_max_disp_seen_mean": float(np.mean([safe_float(r.get("breakaway_max_disp_seen"), 0.0) for r in rs]))}
635:    fieldnames = ["condition", "visible_seed", "policy", "trial_id", "success", "final_fraction", "final_curve", "final_chamfer_to_free", "num_steps", "branch_label", "recoverability_params", "hidden_contact_meta", "breakaway_released", "breakaway_release_step", "breakaway_max_disp_seen", "actions", "candidate_spec", "final_bead_xy", "total_reward", "num_search_trials_evaluated", "search_note", "audit_runtime", "wall_seconds"]
663:        "- `guided_search` is a geometry-guided executable policy used for search sanity.",
676:    parser.add_argument("--policies", nargs="+", default=["nominal", "oracle_pull", "oracle_regrasp", "oracle_wiggle", "oracle_breakaway_then_place", "guided_search"])
```

## task_breakaway_grep

```text
29:      - hidden_breakaway_pin: soft breakaway-style recoverable pin candidate.
43:        "hidden_breakaway_pin",
68:        self._breakaway_released = False
69:        self._breakaway_release_step = None
106:        self._breakaway_released = False
107:        self._breakaway_release_step = None
128:        self._maybe_update_breakaway()
130:        self._maybe_update_breakaway()
277:        if condition == "hidden_breakaway_pin":
278:            return self._apply_hidden_breakaway_pin()
338:        if self.hidden_condition != "hidden_breakaway_pin":
340:        self.hidden_contact_meta["breakaway_released"] = bool(self._breakaway_released)
341:        self.hidden_contact_meta["breakaway_release_step"] = self._breakaway_release_step
344:    def _maybe_update_breakaway(self) -> None:
345:        if self.hidden_condition != "hidden_breakaway_pin":
347:        if self._breakaway_released:
373:            self._breakaway_released = True
374:            self._breakaway_release_step = int(self._ccda_step_count)
377:    def _apply_hidden_breakaway_pin(self) -> None:
407:        self._breakaway_released = False
408:        self._breakaway_release_step = None
411:        params = self._base_recoverability_params("hidden_breakaway_pin")
422:                "condition": "hidden_breakaway_pin",
432:                "breakaway_released": False,
433:                "breakaway_release_step": None,
488:            condition = "hidden_breakaway_pin"
517:                "breakaway_released": False,
```

## selector_grep

```text
10:CONFIG_ENVS = {
169:    selected_env = CONFIG_ENVS.get(selected["label"], {}) if selected else {}
174:        "selected_recoverable_config": selected["label"] if selected and verdict == "PASS" else None,
175:        "selected_recoverable_env": selected_env if selected and verdict == "PASS" else {},
188:            "oracle_success_min": 0.50,
207:        f"- Selected recoverable config: `{payload['selected_recoverable_config']}`",
208:        f"- Selected env vars: `{json.dumps(payload.get('selected_recoverable_env', {}), sort_keys=True)}`",
249:    if payload.get("selected_recoverable_env"):
251:        for k, v in sorted(payload["selected_recoverable_env"].items()):
262:            f"- selected recoverable config: `{payload['selected_recoverable_config']}`\n\n"
264:            + "".join(f"- `{k}={v}`\n" for k, v in sorted(payload["selected_recoverable_env"].items()))
```

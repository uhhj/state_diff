# Phase 0L Resume1 — Routing-Gate Geometry Provenance

- Verdict: `HIDDEN_ROUTING_GATE_RESUME1_SMOKE_BLOCKED`
- Failure class: `engineering_preflight_blocker`
- Provenance seed: `71001`
- Provenance passed: `False`
- Original candidate rejections: `workspace: 4`
- Original maximum required width: `0.098931884335265 m`
- Original configured width: `0.340 m`
- Fixed configured width: `0.350 m`
- Fixed accepted candidates: `0`
- Fixed minimum coverage margin: `0.125534057832368 m`
- Fixed minimum workspace margin: `-0.120534861239818 m`
- Scientific smoke run performed: `False`
- Scientific status: `UNTESTED`
- Training/search performed: `False`

The one-shot seed-71001 provenance capture contradicted the preregistered
coverage diagnosis. The real settled cable was compact and already fit within
the original barrier width; all four candidates failed only the unchanged
workspace predicate. Per protocol, the three-seed smoke was not run and no
additional geometry repair or search was attempted.

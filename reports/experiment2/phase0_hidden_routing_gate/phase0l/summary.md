# Phase 0L — Hidden Routing-Gate Smoke

- Candidate: `hidden_routing_gate_v1_fixed_pull_v1`
- Topology: `hidden_routing_gate_v1`
- Official outcome: `pulled_endpoint_target_success_gap`
- Official gate: `median routing_success_gap >= 1.0`
- Verdict: `HIDDEN_ROUTING_GATE_SMOKE_BLOCKED`
- Failure class: `engineering_blocker`
- Failure: `no legal fixed hidden routing-gate orientation and endpoint`
- Completed pairs: `0`
- Raw files: `0`
- Retry performed: `False`
- Geometry/action/outcome search performed: `False`
- Training performed: `False`

The fixed geometry failed during environment reset for the first requested seed,
before any free/hidden pair completed. No physical-mechanism, routing-outcome,
sensor, vision, or Oracle metric is available. The geometry was not changed and
the simulation was not retried.

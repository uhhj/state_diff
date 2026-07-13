# Phase3.14b-r2.4.2 Frozen-Prior / Factorized-Width Pilot

## Verdict

- Verdict: `PASS` (diagnostic pilot completed, not formal repair)
- Root cause: `phase314b_r242_prior_gradient_isolation_supported`
- Secondary mechanism: `None`
- Train-only recommendation: `frozen_p512_r512`
- Selected configuration: `None`
- Next: `run a separate train-only ordered-geometry pilot using the factorized frozen/decoupled prior recipe; formal validation remains blocked`

## Corrected r2.4.1 interpretation

- Classifier precedence bug supported: `True`
- Corrected primary hypothesis: `x0_prior_gradient_coupling_and_drift`

## Direct prior width stability

- Width 1024: pass seeds `1/3`, stable 2/3 = `False`
- Width 512: pass seeds `1/3`, stable 2/3 = `False`

## Unique-free variants

- `decoupled_p1024_r512`: pass=`True`, drift=`1.5588`, source fraction=`1.000`, z MSE=`7.4041e-06`
- `frozen_p1024_r1024`: pass=`True`, drift=`1`, source fraction=`1.000`, z MSE=`1.26016e-05`
- `frozen_p1024_r512`: pass=`True`, drift=`1`, source fraction=`1.000`, z MSE=`1.15003e-06`
- `frozen_p512_r1024`: pass=`True`, drift=`1`, source fraction=`1.000`, z MSE=`5.94236e-06`
- `frozen_p512_r512`: pass=`True`, drift=`1`, source fraction=`1.000`, z MSE=`1.82966e-06`
- `joint_p1024_r512_control`: pass=`False`, drift=`11862.8`, source fraction=`0.250`, z MSE=`0.0201145`

## Paired low/mid-noise variants

- `decoupled_p1024_r512`: combined pass=`True`, own-target closer=`1.000`, branch cosine p50=`0.999`
- `frozen_p1024_r1024`: combined pass=`False`, own-target closer=`0.995`, branch cosine p50=`0.998`
- `frozen_p1024_r512`: combined pass=`True`, own-target closer=`1.000`, branch cosine p50=`0.997`
- `frozen_p512_r1024`: combined pass=`False`, own-target closer=`0.997`, branch cosine p50=`0.999`
- `frozen_p512_r512`: combined pass=`True`, own-target closer=`1.000`, branch cosine p50=`0.997`

## Boundaries

- Validation targets used: `False`
- Formal test read: `False`
- Formal training: `False`
- Candidate selected: `False`
- Checkpoint saved: `False`
- IDM / candidate execution: `False`
- Phase4 / CPS: `False`

A PASS verdict means only that the train-only diagnostic pilot completed.

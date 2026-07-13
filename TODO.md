# TODO — nanohub-padre

Last updated: 2026-07-13.

The 2026-07-13 device-accuracy review found 24 issues (wrong/missing physics
models, invalid solve sequencing, unphysical defaults, mesh problems); **all
are fixed** — see *Completed* below for the summary and the git history for
details. What remains is the work that needs a PADRE executable or is a
lower-priority enhancement.

---

## Outstanding

### 1. Run the physics validation suite on nanoHUB 🔴 (blocks final sign-off)
`tests/test_physics_validation.py` (7 tests) runs PADRE and checks physics
numbers: diode ideality factor and rectification, MOSFET on/off ratio and
drain-current saturation, BJT β, MOS-cap Cox and HF Cmin. It auto-skips
without a `padre` binary on PATH, so it has **never executed** — run it in a
nanoHUB workspace (or anywhere PADRE is installed):

```bash
python3 -m pytest tests/test_physics_validation.py -v
```

Expected follow-ups: tune ramp step sizes if any solve fails to converge, and
confirm the solar-cell Gaussian emitter (`junction=`, `peak=0`) produces the
intended junction depth (plot the `dop` profile).

### 2. Re-run and sync the example notebooks (needs PADRE)
`examples/notebooks/*.ipynb` embed outputs generated before the accuracy
fixes; results will change (mostly improve) on re-run:

- **04_MOSFET**: cells pass explicit geometry (L=0.1, tox=5 nm, channel 1e18)
  so they still run, but currents change with the new conmob/fldmob/gatmob
  models and Klaassen material card — regenerate outputs and check the prose.
- **06_MOS_Capacitor**: several cells pass `silicon_thickness=0.05` with
  `substrate_doping=1e17` → these now emit the depletion-truncation warning
  (Wd,max ≈ 0.105 µm > 0.05 µm). Decide per cell: switch to the 5 µm bulk
  default, or keep the thin-body demo and acknowledge the warning in prose.
  Cells without `silicon_thickness` now get 5 µm silicon and 1 ns lifetimes —
  the LF C-V should finally reach equilibrium inversion; verify and update text.
- **07_MESFET**: passes no substrate args → now gets the semi-insulating
  n-type 1e10 substrate (reference behavior). Pinch-off and output curves
  change; regenerate and check prose.
- **05_BJT**: β and Gummel plots change with per-region lifetimes and
  surf-rec contacts; regenerate.

### 3. Enhancements (nice-to-have)
- **Golden-deck mesh coverage**: `tests/test_golden_reference.py` compares
  models/materials/doping against `rappture/*.xml`; mesh-refinement ratios
  are asserted separately in `tests/test_accuracy_fixes.py` — could be
  unified against the reference Y.M/X.M lines.
- **MCP/tool descriptions**: `nanohub_tool_description.html` and the
  `nanohub-mcp` / `examples/padre-mcp` side projects may embed old defaults;
  `devices/describe.py` (the source of truth) is synced — regenerate those
  artifacts from it if they are still in use.
- **`examples/mesfet.in` / `mesfet.py` fidelity pair**: now documented as a
  legacy deck-translation example (header notes added); optionally
  regenerate the pair from the 55×43 reference deck and update
  `tests/test_mesfet.py` if a faithful reference translation is preferred.

---

## Completed (2026-07-13)

All 24 findings of the accuracy review are implemented and regression-tested
(226 tests pass locally; 7 physics tests await PADRE, see item 1):

- **Numerics** — `devices/_common.py`: stepped bias ramps (never single
  jumps; ≤0.1 V for forward junctions), PREV/PROJ guess selection,
  ~2500-node mesh check. `generate_deck()` warns on invalid `SOLVE PROJ`.
  `LOG` is emitted after pre-bias ramps so I-V files hold only sweep points.
- **MOSFET** — reference device defaults (L=0.15 µm, tox=2 nm, channel 1e18,
  53×46), conmob/fldmob/gatmob + Klaassen/vsat material cards, y-mesh
  refined *into* the Si/SiO₂ interface, analytic-Vt sweep warning. The old
  defaults produced a device that never turned on.
- **BJT** — base electrode inset from both junctions, mesh refined into both
  junctions, per-region lifetime materials (emat/bmat/cmat), surf-rec
  contacts with VSURFN=0 (VSURFP=0 for PNP) on the base, 60×30 mesh, guards.
- **MESFET** — semi-insulating same-type substrate (1e10) like the reference
  (`substrate_type="opposite"` restores junction isolation), gate electrode
  clamped clear of the n⁺ regions, Fermi statistics, Caughey vsat material,
  55×43 mesh with half the rows in the channel.
- **MOS-cap** — 5 µm / 1e16 single-gate defaults (0.03 µm kept for
  double-gate), 1 ns lifetimes, Wd,max warning, gate ramped before the HF
  sweep and ramped back before the LF sweep.
- **PN diode / Schottky / solar cell** — PREV-first sequencing, reverse
  ramp-back, Schottky thermionic-emission BC (vsurfn/vsurfp) by default,
  mesh finest at the Schottky contact, solar `device_z_width` + surface-
  pinned Gaussian, pn-diode y-width no longer conflated with mesh z-width.
- **`device_z_width` on every factory** — MOSFET/BJT/MESFET/MOS-cap/Schottky
  now expose the z-depth for absolute-current/capacitance scaling (default
  1.0 keeps decks byte-identical: `width=` is only emitted when non-default).
- **Public analytic estimates** — `nanohubpadre.estimate_mosfet_vt()` and
  `nanohubpadre.max_depletion_width_um()` (new `nanohubpadre/estimates.py`);
  the factories' sanity warnings use the same functions.
- **Docs/metadata** — `describe.py`, `schematics.py`, `docs/*.rst`, README
  labels, and I-V/C-V unit documentation (A/µm, F/µm) synced; legacy
  `examples/mesfet.in`/`mesfet.py` pair labeled as deck-translation examples.
- **Tests** — `test_accuracy_fixes.py` (29 deck-level),
  `test_golden_reference.py` (12, compared against `rappture/*.xml`),
  `test_physics_validation.py` (7, PADRE-gated).

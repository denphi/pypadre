# TODO — nanohub-padre

Last updated: 2026-09-06.

The 2026-07-13 device-accuracy review found 24 issues (wrong/missing physics
models, invalid solve sequencing, unphysical defaults, mesh problems); **all
are fixed** — see *Completed* below for the summary and the git history for
details. What remains is the work that needs a PADRE executable or is a
lower-priority enhancement.

---

## Outstanding

### 0. PN-diode review of 2026-09-06 — implemented and verified on nanoHUB ✅
Ran PADRE 2.4E-r15 on nanoHUB (via the mcp4jupyter MCP endpoint) and fixed
what it turned up.  Verified against the `pntoy` reference log
(`OutputLog.dat`): `create_pn_diode(length=9, junction_position=1/3,
p_doping=2e15, n_doping=1e15, taun0=taup0=1e-10)` now reproduces the
reference I(0.6 V) = 1.42066e-07 A to four significant figures (1.4202e-07 A).

Fixed:
- **Mesh** — a single `RATIO=0.8` over 99 intervals drove the junction cell to
  3.2e-11 um (PADRE applies RATIO literally; confirmed against its own mesh
  output) and made `forward_sweep` + `reverse_sweep` abort with
  "Bias stack (10 intermediate points) exceeded".  Replaced with
  `_common.junction_x_mesh()`, which sizes a uniform fine window from
  `estimates.pn_depletion_width_um()` and grades outward with a bounded
  ratio.  The default sweep now completes to -5 V with no PADRE warnings and
  the same ideality (1.031).
- **Sweep arithmetic** — `int(abs(v_end-v_start)/abs(v_step))` truncated
  binary float error, so `(0, 0.6, 0.05)` silently stopped at 0.55 V.  Now
  `_common.sweep_steps()`, which also takes the step's sign from the sweep
  direction and warns on a non-integer step count.
- **Geometry validation** — `junction_position + intrinsic_width > length`
  emitted mesh nodes above `nx`, and PADRE aborted with "Illegal or ambiguous
  mesh line defn!".  Now a `ValueError` up front.
- **Sweep polarity** — electrode 2 is the N side, so a positive
  `forward_sweep` there reverse-biases the diode and logs the ~1e-16 A
  leakage floor as "fwd" (this is what produced the old `examples/iv`, which
  read out as ideality 18).  Now warned about.
- **`Simulation.log`** — `sim.log = Log(...)` was a silent no-op, so
  `examples/pn_diode.py` and `examples/mosfet.py` produced no I-V file at
  all.  Added a real property.
- **Continuity check** — `IVData.continuity_error()` /
  `check_continuity()`.  Terminal currents must satisfy I1 = -I2; the
  reference run is exact at every biased point, the bad run was 6.9% off.
  Above ~1% the current is at the solver's noise floor (~1e-16 A for the
  default device) and ideality/I0 extracted from it is meaningless.
- **Solution files** — stepped solves now set `NO.APPEND`; a default run
  wrote 31 solution files into the working directory and now writes 7.
- **`log_physics_at`** — validated (must start at 0.0, strictly
  increasing) and steps larger than `JUNCTION_STEP` are split.
- **Notebook 02** — the ideality cell referenced undefined `V_sim`/`I_sim`
  inside a bare `except`, so it always printed "Could not extract".  It now
  loads the I-V log, excludes noise-floor points via the continuity check,
  and explains why the default gives n ~ 1.

Regression tests: `tests/test_pn_diode_fixes.py` (39 tests).

Still open from that review, deliberately not changed:
- **Defaults are a short-base diode.** With `length=1.0` and
  `taun0=taup0=1e-6` the diffusion length is ~30 um against ~0.43 um of
  quasi-neutral silicon, so the current is set by the ohmic contacts and
  the measured ideality is 1.03 (1.44 at 1e-9 s, 1.69 at 1e-10 s).  The
  docstring now says so; changing the defaults would move every existing
  user's numbers, so it needs a decision.
- **`ni` is not pinned.**  `pntoy` sets `EG300/NC300/NV300/PERMITTIVITY`
  (ni = 1.008e10); the library uses PADRE's built-ins and notebook 02's hand
  analysis assumes 1.5e10 -> Vbi differs by ~21 mV and I0 by 2.2x.



### 0b. All-device review of 2026-09-06 — implemented and verified ✅
Ran all seven factories on PADRE 2.4E-r15 on nanoHUB.  After the fixes below,
every device completes its requested sweep with **zero PADRE warnings**:

| device | sweep reaches | peak \|I\| | note |
|---|---|---|---|
| pn_diode | -5.0 V | 1.06e-05 A | |
| mosfet Id-Vd | 1.5 V | 4.25e-04 A | was stalling at 0.093 V |
| mosfet Id-Vg | 1.5 V | 6.78e-05 A | was 16 cut-back fragments |
| mesfet | 3.0 V | 5.34e-04 A | |
| bjt Gummel | 0.8 V | 2.21e-05 A | beta 63->85->69 over 9 decades |
| mos_capacitor | 3.0 V | Cox 1.69e-14 F | Cmin 3.24e-16 F |
| schottky | -2.0 V | 3.54e-05 A | ~1e6 rectification |

Fixed:
- **Sweep endpoints in every factory** — the `int()` truncation fixed in
  `pn_diode` was still live at nine sites (bjt x2, mesfet, mos_capacitor,
  schottky x2, mosfet x2, solar_cell).  Observed live: `create_solar_cell(
  forward_sweep=(0.0, 0.7, 0.1))` stopped at 0.6 V.  All now use
  `_common.sweep_steps()`.
- **MOSFET y-mesh** — fixed ratios 0.875 then 0.7 compounded to
  h_min = 8e-6 um (0.08 A) at the Si/SiO2 interface, and PADRE stalled the
  drain sweep at Vd = 0.093 V of a requested 1.5 V.  Proven by a controlled
  experiment: relaxing only those ratios completed the sweep.  Ratios are now
  computed from a target interface cell size via `_common.contract_ratio()` /
  `expand_ratio()`; h_min = 3.0e-4 um.  Grading *directions* are unchanged, so
  the reference-deck assertions still hold.
- **MOS-cap mesh** — `ny_oxide` defaulted to 100 lines across a 2 nm oxide
  (0.02 nm cells) and RATIO=0.8 over 50 intervals reached h_min = 3.6e-9 um
  (3.6 femtometres), the worst mesh in the library.  Default is now 12, the
  near-interface zone is sized from `ny_silicon` instead of `ny_oxide`, and an
  over-meshed oxide warns.  C-V is preserved: Cox 1.6907e-14 F (analytic
  1.7250e-14, -2.0%), Cmin 3.2400e-16 F (analytic 3.3979e-16, -4.6%).
- **MESFET mesh** — grading bounded; the spurious zero-bias current dropped
  from 7.9e-6 A to 3.3e-6 A (1.3% -> 0.6% of on-current).  Documented as the
  error floor in the factory docstring: on-state Id-Vds is trustworthy,
  pinch-off and subthreshold are not.
- **Solar-cell warning** — it told users to raise `device_z_width` to lift
  currents above the noise floor.  That cannot work, and does not: verified
  z=1 and z=1e4 give currents identical to 4 digits after scaling.  The
  warning now says so and tells callers not to read I-V from this device.

Regression tests added to `tests/test_pn_diode_fixes.py`: sweep endpoints and
sub-atomic-cell checks parametrised over all eight factory/sweep combinations,
plus MOSFET and MOS-cap mesh regressions.  `tests/test_golden_reference.py`
(12 tests against `rappture/*.xml`) and the mesh-direction assertions in
`tests/test_accuracy_fixes.py` (29 tests) still pass unchanged.

Still open: the MESFET 0.6% equilibrium floor, and the solar cell, whose dark
current PADRE cannot resolve at any device size.

### 0c. NIN / PIP isotype diode factory — added 2026-09-06 ✅
`create_nin_diode()` (aliases `nin_diode`, `pip_diode`) builds a
contact / high-resistivity barrier / contact structure with no PN junction:
n+/n-/n+ for `device_type="nin"`, p+/p-/p+ for `"pip"`.  Registered in
`devices/__init__.py`, the top-level package, `describe.py` and
`schematics.py`.

Because both contacts are the same type the device must not rectify, so the
defining check is I-V antisymmetry.  Measured on PADRE 2.4E-r15 (nanoHUB),
sweeping -2 V to +2 V with the defaults:

- exit 0, no convergence warnings
- terminal-current continuity **0.000%** at every biased point
- equilibrium current 4e-18 A
- **I-V antisymmetry within 0.001%**: |I(+2 V)| = 7.2627e-05 A against
  |I(-2 V)| = 7.2626e-05 A
- PIP gives 3.1e-05 A at the same bias, ~2.3x lower, tracking the
  electron/hole mobility ratio

Transport regime at 2 V: ~13x the simple ohmic estimate for the 1e14 cm^-3
barrier but ~1.7x below the trap-free Mott-Gurney limit, i.e. the
injection-enhanced regime between ohmic and full SCLC — as expected once the
injected density exceeds the background doping.

Mesh refinement is sized from the new `estimates.debye_length_um()` (there is
no depletion region to measure), and both isotype interfaces land exactly on
mesh nodes.

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

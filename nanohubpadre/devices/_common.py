"""
Shared helpers for device factory functions.

These utilities enforce the numerical-setup rules that the reference
nanoHUB Rappture decks follow:

- ``SOLVE PROJ`` needs two prior solutions; the first bias solve after
  ``SOLVE INIT`` must use ``PREV``.
- Large bias changes are applied as stepped ramps, never as a single jump.
- Total mesh size must stay below the nanoHUB PADRE node limit
  (~2500 nodes) or the solver fails with "Insufficient storage for real
  numerical factorization".
"""

import math
import warnings
from typing import Dict, Optional

from ..solver import Solve

# PADRE on nanoHUB fails above roughly this many mesh nodes.
NANOHUB_NODE_LIMIT = 2500

# Default maximum bias step sizes (volts) for ramps.
# Forward-biased junctions (BJT base-emitter) need small steps;
# MOS gates and reverse-biased/drain-side contacts tolerate larger ones.
JUNCTION_STEP = 0.1
GATE_STEP = 0.2
DRAIN_STEP = 0.2


def check_mesh_size(nx: int, ny: int, device: str = "device") -> None:
    """Warn when the mesh exceeds the nanoHUB PADRE node limit."""
    nodes = nx * ny
    if nodes > NANOHUB_NODE_LIMIT:
        warnings.warn(
            f"{device}: mesh has {nx}x{ny} = {nodes} nodes, above the "
            f"~{NANOHUB_NODE_LIMIT}-node limit of PADRE on nanoHUB "
            f"(fails with 'Insufficient storage for real numerical "
            f"factorization'). Reduce nx and/or ny.",
            UserWarning,
            stacklevel=3,
        )


def solve_guess(n_prior: int) -> Dict[str, bool]:
    """
    Initial-guess flags for the next SOLVE.

    PADRE's ``PROJ`` guess extrapolates from the two most recent
    solutions; with fewer than two available, ``PREV`` must be used.
    """
    if n_prior < 2:
        return {"previous": True}
    return {"project": True}


def add_bias_ramp(
    sim,
    electrode: int,
    v_from: float,
    v_to: float,
    n_prior: int,
    max_step: float = GATE_STEP,
    outfile: Optional[str] = None,
) -> int:
    """
    Ramp `electrode` from `v_from` to `v_to` in steps of at most `max_step`.

    Mirrors the reference Rappture decks, which never jump to a bias in a
    single solve (e.g. the MOS-cap tool ramps to the sweep start in 100
    steps before enabling the AC log).

    Parameters
    ----------
    sim : Simulation
        Simulation to append the solve to.
    electrode : int
        Electrode number (1-9).
    v_from : float
        Bias the electrode is currently at (last solved value).
    v_to : float
        Target bias.
    n_prior : int
        Number of solutions computed so far (1 right after ``SOLVE INIT``).
    max_step : float
        Maximum voltage step per solve.
    outfile : str, optional
        Solution output file for the ramp.

    Returns
    -------
    int
        Updated prior-solution count.
    """
    delta = v_to - v_from
    if abs(delta) < 1e-12:
        return n_prior

    nsteps = max(1, math.ceil(abs(delta) / max_step - 1e-9))
    vstep = round(delta / nsteps, 12)

    kwargs = solve_guess(n_prior)
    kwargs[f"v{electrode}"] = v_from
    sim.add_solve(Solve(
        vstep=vstep,
        nsteps=nsteps,
        electrode=electrode,
        no_append=True,
        outfile=outfile,
        **kwargs,
    ))
    return n_prior + nsteps + 1


# ---------------------------------------------------------------------------
# Sweep arithmetic
# ---------------------------------------------------------------------------

def sweep_steps(v_start: float, v_end: float, v_step: float,
                what: str = "sweep") -> tuple:
    """
    Number of VSTEP increments and the true endpoint for a bias sweep.

    PADRE solves the starting bias and then takes NSTEPS further
    increments, so a sweep covers ``nsteps + 1`` points and ends at
    ``v_start + v_step * nsteps``.

    Uses rounding rather than truncation: ``int(0.6 / 0.05)`` is 11, not
    12, because 0.6/0.05 evaluates to 11.999999999999998 in binary
    floating point, which silently stopped sweeps one step short of the
    requested endpoint.

    The magnitude of ``v_step`` is used and its sign is taken from
    ``v_end - v_start``, so a step whose sign disagrees with the sweep
    direction no longer runs the sweep backwards.

    Returns
    -------
    (nsteps, signed_step, v_final)
    """
    span = v_end - v_start
    if v_step == 0:
        raise ValueError(f"{what}: v_step must be non-zero")
    if span == 0:
        raise ValueError(
            f"{what}: v_start and v_end are both {v_start}; nothing to sweep")

    step = math.copysign(abs(v_step), span)
    if math.copysign(1.0, v_step) != math.copysign(1.0, span):
        warnings.warn(
            f"{what}: v_step={v_step} disagrees in sign with the sweep "
            f"{v_start} -> {v_end}; using {step} so the sweep runs in the "
            f"requested direction.",
            UserWarning, stacklevel=3,
        )

    exact = abs(span) / abs(step)
    nsteps = int(round(exact))
    if abs(exact - nsteps) > 1e-6:          # genuinely not a whole number
        nsteps = max(1, int(math.floor(exact)))
        warnings.warn(
            f"{what}: {v_start} -> {v_end} is not a whole number of "
            f"{abs(v_step)} V steps; stopping at "
            f"{v_start + step * nsteps:g} V.",
            UserWarning, stacklevel=3,
        )
    nsteps = max(1, nsteps)
    return nsteps, round(step, 12), round(v_start + step * nsteps, 12)


# ---------------------------------------------------------------------------
# Junction-aware mesh generation
# ---------------------------------------------------------------------------

# PADRE applies X.MESH/Y.MESH RATIO literally as a geometric progression
# across the sub-intervals of a segment (verified against PADRE's own mesh
# output).  A ratio near the manual's 0.667-1.5 bounds compounds: r=0.8 over
# 99 intervals spans nine orders of magnitude and drives the cell at the
# junction to ~3e-11 um.  Keep the per-segment ratio well inside that.
MAX_MESH_RATIO = 1.35


def _segment_ratio(seg_length: float, n_intervals: int, h_end: float,
                   max_ratio: float = MAX_MESH_RATIO) -> float:
    """
    Ratio for a graded segment whose *finest* cell is ``h_end``.

    Solves ``h_end * (r**n - 1) / (r - 1) == seg_length`` for r >= 1 and
    clamps to ``max_ratio``.  Returns 1.0 when the segment cannot be made
    coarser than ``h_end`` (i.e. it is already fine enough).
    """
    if n_intervals <= 1 or seg_length <= 0 or h_end <= 0:
        return 1.0
    if seg_length <= n_intervals * h_end:
        return 1.0                       # uniform is already fine enough
    lo, hi = 1.0 + 1e-9, max_ratio
    if h_end * (hi ** n_intervals - 1) / (hi - 1) < seg_length:
        return max_ratio                 # clamped: cannot reach with max_ratio
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if h_end * (mid ** n_intervals - 1) / (mid - 1) < seg_length:
            lo = mid
        else:
            hi = mid
    return round(0.5 * (lo + hi), 4)


def _outer_intervals(seg_length: float, h: float, ratio: float) -> int:
    """Intervals needed to span ``seg_length`` growing from ``h`` by ``ratio``."""
    if seg_length <= 0 or h <= 0:
        return 0
    return max(1, math.ceil(math.log1p(seg_length * (ratio - 1) / h)
                            / math.log(ratio)))


def junction_x_mesh(mesh, length: float, junctions, nx: int,
                    fine_half_width: float, growth: float = 1.15):
    """
    Emit X.MESH lines that resolve one or more junctions properly.

    A uniform "fine window" spans from ``first junction - fine_half_width``
    to ``last junction + fine_half_width``; every junction inside it is a
    mesh line, so region and doping boundaries land exactly on nodes.  The
    rest of the device is graded outward from the same cell size at a
    bounded ratio, so cell size varies over a small factor instead of the
    nine orders of magnitude a single aggressive RATIO produces.

    The node budget is solved for self-consistently: the fine cell size is
    chosen so that the fine window plus the graded shoulders use exactly
    ``nx`` lines, which stops a short shoulder from hoarding nodes.

    Parameters
    ----------
    mesh : Mesh
        Mesh to append X.MESH lines to.
    length : float
        Device length in microns.
    junctions : sequence of float
        Junction x-coordinates in microns.
    nx : int
        Total number of x mesh lines.
    fine_half_width : float
        How far the uniform fine window extends past the outer junctions.
    growth : float
        Target cell growth ratio in the graded shoulders.

    Returns
    -------
    dict
        ``{junction_x: node_index}`` for each junction, plus ``"h_fine"``.
    """
    js = sorted(float(j) for j in junctions)
    a = max(0.0, js[0] - fine_half_width)
    b = min(length, js[-1] + fine_half_width)
    fine_len, left_len, right_len = b - a, a, length - b

    inner = sorted(set(round(v, 12)
                       for v in [a] + [j for j in js if a < j < b] + [b]))
    n_break = len(inner) - 1                      # fine sub-segments

    # Solve for the fine cell size h that makes the total line count nx.
    def total(h):
        return (max(fine_len / h, n_break) + 1
                + _outer_intervals(left_len, h, growth)
                + _outer_intervals(right_len, h, growth))

    lo, hi = length / (nx * 50.0), length
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if total(mid) > nx:
            lo = mid
        else:
            hi = mid
    h = math.sqrt(lo * hi)

    n_left = _outer_intervals(left_len, h, growth)
    n_right = _outer_intervals(right_len, h, growth)
    n_fine = nx - 1 - n_left - n_right
    if n_fine < n_break:                          # degenerate; fall back
        n_fine = max(n_break, nx - 1 - n_break)
        n_left = min(n_left, max(0, nx - 1 - n_fine))
        n_right = max(0, nx - 1 - n_fine - n_left)
    h_fine = fine_len / n_fine if n_fine else h

    node = 1
    mesh.add_x_mesh(1, 0.0, ratio=1)
    if left_len > 0 and n_left > 0:
        node += n_left
        r = _segment_ratio(left_len, n_left, h_fine)
        mesh.add_x_mesh(node, round(a, 10), ratio=round(1.0 / r, 4))

    nodes_at = {}
    spans = [inner[i + 1] - inner[i] for i in range(n_break)]
    rem = n_fine
    for i, s in enumerate(spans):
        k = rem - (n_break - 1 - i) if i == n_break - 1 else \
            max(1, min(int(round(n_fine * s / fine_len)), rem - (n_break - 1 - i)))
        node += k
        rem -= k
        mesh.add_x_mesh(node, round(inner[i + 1], 10), ratio=1)
        nodes_at[round(inner[i + 1], 12)] = node

    if node != nx:
        r = _segment_ratio(right_len, nx - node, h_fine) if right_len > 0 else 1
        mesh.add_x_mesh(nx, round(length, 10), ratio=r)

    out = {j: nodes_at.get(round(j, 12)) for j in js}
    out["h_fine"] = h_fine
    return out


def contract_ratio(seg_length: float, n_intervals: int, h_fine: float,
                   max_ratio: float = MAX_MESH_RATIO) -> float:
    """
    RATIO for a segment whose cells shrink toward its far end (<= 1).

    Use when the feature to resolve (an interface, a junction) sits at the
    end of the segment.  ``h_fine`` is the cell size wanted there; the
    ratio is bounded so the grading cannot compound into sub-atomic cells.
    """
    return round(1.0 / _segment_ratio(seg_length, n_intervals, h_fine,
                                      max_ratio), 4)


def expand_ratio(seg_length: float, n_intervals: int, h_fine: float,
                 max_ratio: float = MAX_MESH_RATIO) -> float:
    """
    RATIO for a segment whose cells grow away from its near end (>= 1).

    The mirror of :func:`contract_ratio`, for when the feature sits at the
    start of the segment.
    """
    return _segment_ratio(seg_length, n_intervals, h_fine, max_ratio)

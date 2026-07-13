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
        outfile=outfile,
        **kwargs,
    ))
    return n_prior + nsteps + 1

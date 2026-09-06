"""
PN Junction Diode factory function.
"""

import math
import warnings
from typing import List, Optional, Tuple
from ..simulation import Simulation
from ..mesh import Mesh
from ..region import Region
from ..electrode import Electrode
from ..doping import Doping
from ..contact import Contact
from ..material import Material
from ..models import Models
from ..solver import System, Solve
from ..options import Options
from ..log import Log
from ..estimates import pn_depletion_width_um
from ._common import (check_mesh_size, solve_guess, add_bias_ramp, GATE_STEP,
                      JUNCTION_STEP, sweep_steps, junction_x_mesh)


def create_pn_diode(
    # Geometry parameters
    length: float = 1.0,
    width: float = 1.0,
    device_z_width: float = 1.0,
    junction_position: float = 0.5,
    intrinsic_width: float = 0.0,
    # Mesh parameters
    nx: int = 200,
    ny: int = 3,
    # Doping parameters
    p_doping: float = 1e17,
    n_doping: float = 1e17,
    intrinsic_doping: float = 1.5e10,
    # Physical models
    temperature: float = 300,
    srh: bool = True,
    conmob: bool = True,
    fldmob: bool = True,
    impact: bool = False,
    # Material parameters
    taun0: float = 1e-6,
    taup0: float = 1e-6,
    # Simulation options
    title: Optional[str] = None,
    postscript: bool = False,
    # Output logging options
    log_iv: bool = False,
    iv_file: str = "iv",
    log_bands_eq: bool = False,
    log_bands_bias: bool = False,
    # Voltage sweep options
    forward_sweep: Optional[Tuple[float, float, float]] = None,
    reverse_sweep: Optional[Tuple[float, float, float]] = None,
    sweep_electrode: int = 1,
    # Physics-profile logging
    log_physics_at: Optional[List[float]] = None,
) -> Simulation:
    """
    Create a PN junction diode simulation.

    Creates a 1D-like PN diode structure with configurable doping profiles,
    mesh refinement, and physical models.

    Parameters
    ----------
    length : float
        Total device length in microns (default: 1.0)
    width : float
        Device width (y extent) in microns (default: 1.0)
    device_z_width : float
        Depth of the device in the third dimension in microns (default:
        1.0). Terminal currents scale linearly with this value. Note:
        earlier versions passed `width` here, silently scaling currents
        with the y extent squared.
    junction_position : float
        Position of the P-I (or P-N) junction as fraction of length (default: 0.5)
    intrinsic_width : float
        Width of the intrinsic (lightly-doped) region in microns (default: 0.0).
        When 0, structure is a standard PN diode.  When > 0, an intrinsic
        region is inserted between P and N, forming a PIN diode.
    nx : int
        Number of mesh points in x direction (default: 200)
    ny : int
        Number of mesh points in y direction (default: 3)
    p_doping : float
        P-type doping concentration in cm^-3 (default: 1e17)
    n_doping : float
        N-type doping concentration in cm^-3 (default: 1e17)
    intrinsic_doping : float
        Intrinsic region doping concentration in cm^-3 (default: 1.5e10).
        Only used when intrinsic_width > 0.
    temperature : float
        Simulation temperature in Kelvin (default: 300)
    srh : bool
        Enable Shockley-Read-Hall recombination (default: True)
    conmob : bool
        Enable concentration-dependent mobility (default: True)
    fldmob : bool
        Enable field-dependent mobility (default: True)
    impact : bool
        Enable impact ionization (default: False)
    taun0 : float
        Electron lifetime in seconds (default: 1e-6)
    taup0 : float
        Hole lifetime in seconds (default: 1e-6)

        Note: with the default 1 um device the minority-carrier diffusion
        length at 1e-6 s is ~30 um, far longer than the ~0.43 um
        quasi-neutral regions, so the diode is short-base and its current
        is set by recombination at the ohmic contacts.  Lifetime barely
        affects the forward I-V until it drops to ~1e-9 s (measured
        ideality: 1.03 at 1e-6 s, 1.44 at 1e-9 s, 1.69 at 1e-10 s).  Use
        1e-9 s or shorter to see depletion-region recombination (n > 1).
    title : str, optional
        Simulation title
    postscript : bool
        Enable PostScript output (default: False)
    log_iv : bool
        If True, add I-V logging (default: False)
    iv_file : str
        Filename for I-V log (default: "iv")
    log_bands_eq : bool
        If True, log band diagrams at equilibrium (default: False)
    log_bands_bias : bool
        If True, log band diagrams during voltage sweeps (default: False)
    forward_sweep : tuple (v_start, v_end, v_step), optional
        If provided, adds a forward bias voltage sweep.
        Example: (0.0, 0.8, 0.05) sweeps from 0V to 0.8V in 0.05V steps
    reverse_sweep : tuple (v_start, v_end, v_step), optional
        If provided, adds a reverse bias voltage sweep.
        Example: (0.0, -5.0, -0.5) sweeps from 0V to -5V in 0.5V steps
    sweep_electrode : int
        Electrode number to apply voltage sweeps (default: 1 = P side /
        anode, so a positive forward_sweep forward-biases the junction;
        2 = N side / cathode, where the polarity is inverted).  A sweep
        whose polarity would reverse-bias a "forward_sweep" is warned
        about: it produces only the ~1e-16 A leakage floor.
    log_physics_at : list of float, optional
        Bias voltages (V) at which to capture physics profiles along the
        device: recombination rate, electric field, electron density, hole
        density, and net carrier density.  When provided the forward sweep
        is replaced by a step-by-step ramp that inserts Plot1D commands
        between each bias point.  The list must start at 0.0 (equilibrium
        is always logged first) and increase strictly; steps larger than
        JUNCTION_STEP (0.1 V) are split into sub-steps so the forward-biased
        junction converges.
        Example: [0.0, 0.2, 0.4, 0.6]

    Returns
    -------
    Simulation
        Configured PN diode simulation ready to run

    Example
    -------
    >>> # Basic PN diode - no solve commands, add your own
    >>> sim = create_pn_diode(length=2.0, p_doping=1e16, n_doping=1e18)
    >>> sim.add_solve(Solve(initial=True))
    >>> print(sim.generate_deck())
    >>>
    >>> # PIN diode with 0.2 um intrinsic region
    >>> sim = create_pn_diode(
    ...     length=1.0, junction_position=0.3, intrinsic_width=0.4,
    ...     p_doping=1e17, n_doping=1e17,
    ...     log_iv=True, forward_sweep=(0.0, 1.0, 0.05)
    ... )
    >>> result = sim.run()
    >>>
    >>> # Complete PN simulation with sweeps and logging
    >>> sim = create_pn_diode(
    ...     log_iv=True,
    ...     log_bands_eq=True,
    ...     log_bands_bias=True,
    ...     forward_sweep=(0.0, 0.8, 0.05),
    ...     reverse_sweep=(0.0, -5.0, -0.5)
    ... )
    >>> result = sim.run()
    >>> sim.plot_band_diagram()  # Plots all logged band diagrams
    """
    is_pin = intrinsic_width > 0.0
    sim = Simulation(title=title or ("PIN Diode" if is_pin else "PN Junction Diode"))
    sim._device_type = "pn_diode"
    sim._device_kwargs = dict(
        length=length, width=width, device_z_width=device_z_width,
        junction_position=junction_position,
        intrinsic_width=intrinsic_width, nx=nx, ny=ny,
        p_doping=p_doping, n_doping=n_doping, intrinsic_doping=intrinsic_doping,
        temperature=temperature, srh=srh, conmob=conmob, fldmob=fldmob,
        impact=impact, taun0=taun0, taup0=taup0, title=title,
        postscript=postscript, log_iv=log_iv, iv_file=iv_file,
        log_bands_eq=log_bands_eq, log_bands_bias=log_bands_bias,
        forward_sweep=forward_sweep, reverse_sweep=reverse_sweep,
        sweep_electrode=sweep_electrode, log_physics_at=log_physics_at,
    )

    # Options
    if postscript:
        sim.options = Options(postscript=True)

    check_mesh_size(nx, ny, "create_pn_diode")

    # Compute junction boundaries
    junction_x = junction_position * length           # P-I (or P-N) boundary
    intrinsic_end = junction_x + intrinsic_width      # I-N boundary (== junction_x when PN)

    # --- Geometry validation -------------------------------------------------
    # Without these, an out-of-range junction silently emits X.MESH lines with
    # node numbers above nx and out of order, which PADRE rejects outright with
    # "Illegal or ambiguous mesh line defn! / PADRE Aborted".
    if length <= 0:
        raise ValueError(f"create_pn_diode: length must be positive, got {length}")
    if not 0.0 < junction_position < 1.0:
        raise ValueError(
            f"create_pn_diode: junction_position must be strictly between 0 and 1, "
            f"got {junction_position}")
    if intrinsic_width < 0:
        raise ValueError(
            f"create_pn_diode: intrinsic_width must be >= 0, got {intrinsic_width}")
    if intrinsic_end >= length:
        raise ValueError(
            f"create_pn_diode: the intrinsic region runs off the end of the device "
            f"(junction at {junction_x:g} um + intrinsic_width {intrinsic_width:g} um "
            f"= {intrinsic_end:g} um >= length {length:g} um). Reduce "
            f"junction_position or intrinsic_width.")
    if nx < 8 or ny < 2:
        raise ValueError(
            f"create_pn_diode: need at least nx=8, ny=2 mesh lines, got nx={nx}, ny={ny}")
    if sweep_electrode not in (1, 2):
        raise ValueError(
            f"create_pn_diode: sweep_electrode must be 1 (P side/anode) or "
            f"2 (N side/cathode), got {sweep_electrode}")

    # --- Sweep polarity ------------------------------------------------------
    # Electrode 1 sits in the P region and electrode 2 in the N region, so a
    # positive bias forward-biases the junction only on electrode 1.  Sweeping
    # a positive "forward_sweep" on electrode 2 reverse-biases the diode and
    # logs the ~1e-17 A leakage floor into a file named "fwd", which then reads
    # out as an ideality factor of ~18.  Catch it instead of producing that.
    def _polarity_ok(sweep, want_forward):
        if sweep is None:
            return True
        span = sweep[1] - sweep[0]
        if span == 0:
            return True
        forward = (span > 0) if sweep_electrode == 1 else (span < 0)
        return forward == want_forward

    if not _polarity_ok(forward_sweep, True):
        warnings.warn(
            f"create_pn_diode: forward_sweep={forward_sweep} on electrode "
            f"{sweep_electrode} ({'P side/anode' if sweep_electrode == 1 else 'N side/cathode'}) "
            f"REVERSE-biases the junction, so the diode never turns on and the "
            f"'fwd' I-V log holds only leakage current. Use sweep_electrode=1 "
            f"with a positive sweep, or sweep_electrode=2 with a negative one.",
            UserWarning, stacklevel=2,
        )
    if not _polarity_ok(reverse_sweep, False):
        warnings.warn(
            f"create_pn_diode: reverse_sweep={reverse_sweep} on electrode "
            f"{sweep_electrode} forward-biases the junction.",
            UserWarning, stacklevel=2,
        )

    # Mesh refined around the junction(s).
    # width= is the z-depth (current scaling), not the y extent.
    sim.mesh = Mesh(nx=nx, ny=ny, width=device_z_width, outfile="mesh")

    # Size the fine window from the physics rather than a fixed RATIO: PADRE
    # applies RATIO literally as a geometric progression, so a single graded
    # segment over ~100 intervals collapses to femtometre cells at the junction
    # and makes forward-then-reverse sweeps fail to converge.
    v_rev = 0.0
    if reverse_sweep is not None:
        v_rev = abs(min(reverse_sweep[0], reverse_sweep[1]))
    w_dep = pn_depletion_width_um(p_doping, n_doping, reverse_bias=v_rev,
                                  temperature=temperature)
    junctions = [junction_x, intrinsic_end] if is_pin else [junction_x]
    room = min(junction_x, length - intrinsic_end)
    fine_half_width = max(min(w_dep, 0.9 * room), 0.05 * room)

    mesh_info = junction_x_mesh(sim.mesh, length, junctions, nx,
                                fine_half_width=fine_half_width)
    nx_pi = mesh_info[junctions[0]]
    nx_in = mesh_info[junctions[-1]]

    sim.mesh.add_y_mesh(1, 0, ratio=1)
    sim.mesh.add_y_mesh(ny, width, ratio=1)

    # Silicon regions
    if is_pin:
        sim.add_region(Region(1, ix_low=1, ix_high=nx_pi, iy_low=1, iy_high=ny, silicon=True))
        sim.add_region(Region(1, ix_low=nx_pi, ix_high=nx_in, iy_low=1, iy_high=ny, silicon=True))
        sim.add_region(Region(1, ix_low=nx_in, ix_high=nx, iy_low=1, iy_high=ny, silicon=True))
    else:
        sim.add_region(Region(1, ix_low=1, ix_high=nx_pi, iy_low=1, iy_high=ny, silicon=True))
        sim.add_region(Region(1, ix_low=nx_pi, ix_high=nx, iy_low=1, iy_high=ny, silicon=True))

    # Electrodes at device ends
    sim.add_electrode(Electrode(1, ix_low=1, ix_high=1, iy_low=1, iy_high=ny))
    sim.add_electrode(Electrode(2, ix_low=nx, ix_high=nx, iy_low=1, iy_high=ny))

    # Doping
    sim.add_doping(Doping(region=1, p_type=True, concentration=p_doping,
                          x_left=0, x_right=junction_x, y_top=0, y_bottom=width, uniform=True))
    if is_pin:
        sim.add_doping(Doping(region=1, n_type=True, concentration=intrinsic_doping,
                              x_left=junction_x, x_right=intrinsic_end,
                              y_top=0, y_bottom=width, uniform=True))
    sim.add_doping(Doping(region=1, n_type=True, concentration=n_doping,
                          x_left=intrinsic_end, x_right=length, y_top=0, y_bottom=width, uniform=True))

    # Contacts - ohmic contacts for all electrodes
    sim.add_contact(Contact(all_contacts=True, neutral=True))

    # Material with lifetimes
    sim.add_material(Material(name="silicon", taun0=taun0, taup0=taup0,
                              trap_type="0", etrap=0))

    # Physical models
    sim.models = Models(srh=srh, conmob=conmob, fldmob=fldmob, impact=impact,
                        temperature=temperature)
    sim.system = System(electrons=True, holes=True, newton=True)

    # I-V logging
    if log_iv:
        sim.add_log(Log(ivfile=iv_file))

    # Line cut position for band diagrams (horizontal through middle of device)
    y_cut = width / 2.0

    # Only add solve commands if sweeps or logging are specified
    if (forward_sweep is not None or reverse_sweep is not None
            or log_bands_eq or log_physics_at is not None):
        # Always start with equilibrium solve
        sim.add_solve(Solve(initial=True, outfile="eq"))
        n_prior = 1     # solutions computed so far
        v_sweep = 0.0   # last solved bias on the sweep electrode

        # Log equilibrium band diagram if requested
        if log_bands_eq:
            sim.log_band_diagram(
                outfile_prefix="eq",
                x_start=0.0, y_start=y_cut,
                x_end=length, y_end=y_cut
            )

        # Physics-profile stepping: step one bias at a time so that
        # Plot1D commands fire at each requested voltage.
        # When active this replaces forward_sweep.
        if log_physics_at is not None:
            if not log_physics_at:
                raise ValueError("create_pn_diode: log_physics_at is empty")
            if abs(log_physics_at[0]) > 1e-12:
                raise ValueError(
                    f"create_pn_diode: log_physics_at must start at 0.0 "
                    f"(equilibrium is always the first snapshot), got "
                    f"{log_physics_at[0]}. Prepend 0.0 to the list.")
            if any(b <= a for a, b in zip(log_physics_at, log_physics_at[1:])):
                raise ValueError(
                    f"create_pn_diode: log_physics_at must be strictly "
                    f"increasing, got {log_physics_at}")
            _cut = dict(x_start=0.0, y_start=y_cut,
                        x_end=length, y_end=y_cut)

            # Equilibrium snapshot
            sim.log_physics("eq", **_cut)

            # Step through remaining bias points one at a time, splitting any
            # step larger than JUNCTION_STEP: a forward-biased junction does
            # not converge across a single large jump.
            v_prev = 0.0
            for v in log_physics_at[1:]:
                tag = f"v{v:.1f}".replace(".", "p")   # e.g. "v0p2"
                dv = v - v_prev
                sub = max(1, math.ceil(abs(dv) / JUNCTION_STEP - 1e-9))
                solve_kwargs = dict(**solve_guess(n_prior),
                                    vstep=round(dv / sub, 12), nsteps=sub,
                                    electrode=sweep_electrode,
                                    no_append=True, outfile=f"sol_{tag}")
                if sweep_electrode == 1:
                    solve_kwargs["v1"] = v_prev
                else:
                    solve_kwargs["v2"] = v_prev
                sim.add_solve(Solve(**solve_kwargs))
                n_prior += sub + 1
                sim.log_physics(tag, **_cut)
                v_prev = v
            v_sweep = v_prev

        # Forward bias sweep (skipped when log_physics_at is used)
        elif forward_sweep is not None:
            v_start, v_end, v_step = forward_sweep
            nsteps, v_step, v_final = sweep_steps(v_start, v_end, v_step,
                                                  "create_pn_diode forward_sweep")
            sim.add_solve(Solve(
                **solve_guess(n_prior),
                v1=v_start if sweep_electrode == 1 else 0.0,
                v2=v_start if sweep_electrode == 2 else 0.0,
                vstep=v_step,
                nsteps=nsteps,
                electrode=sweep_electrode,
                no_append=True,
                outfile="fwd"
            ))
            n_prior += nsteps + 1
            v_sweep = v_final
            if log_bands_bias:
                sim.log_band_diagram(
                    outfile_prefix="fwd",
                    x_start=0.0, y_start=y_cut,
                    x_end=length, y_end=y_cut,
                    include_qf=True
                )

        # Reverse bias sweep
        if reverse_sweep is not None:
            v_start, v_end, v_step = reverse_sweep
            nsteps, v_step, v_final = sweep_steps(v_start, v_end, v_step,
                                                  "create_pn_diode reverse_sweep")

            # A previous sweep left the diode at a different bias: ramp
            # back to the reverse-sweep start instead of jumping.  Coming
            # down off forward bias is the hardest part of the sequence, so
            # step it at JUNCTION_STEP rather than the larger GATE_STEP.
            # The ramp retraces already-logged forward bias points, so the
            # I-V data is unaffected apart from duplicates.
            if abs(v_start - v_sweep) > 1e-10:
                n_prior = add_bias_ramp(sim, sweep_electrode, v_sweep,
                                        v_start, n_prior,
                                        max_step=JUNCTION_STEP,
                                        outfile="rev_ramp")
                v_sweep = v_start

            sim.add_solve(Solve(
                **solve_guess(n_prior),
                v1=v_start if sweep_electrode == 1 else 0.0,
                v2=v_start if sweep_electrode == 2 else 0.0,
                vstep=v_step,
                nsteps=nsteps,
                electrode=sweep_electrode,
                no_append=True,
                outfile="rev"
            ))
            n_prior += nsteps + 1
            v_sweep = v_final
            if log_bands_bias:
                sim.log_band_diagram(
                    outfile_prefix="rev",
                    x_start=0.0, y_start=y_cut,
                    x_end=length, y_end=y_cut,
                    include_qf=True
                )

    return sim


# Alias
pn_diode = create_pn_diode
